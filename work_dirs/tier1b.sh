#!/usr/bin/env bash
# PLAN.md T1-6/T1-7 follow-up + T1-8, as one detached queue.
#   setsid nohup bash work_dirs/tier1b.sh > work_dirs/tier1b-console.log 2>&1 < /dev/null &
#
# Round 1 (work_dirs/tier1-console.log) against the seed-0 control
# work_dirs/t0-seed0-bin0.2 (real AbsRel 0.1773 / d1 0.8082 / TCE 0.0344):
#   warp 0.5   TCE 0.0327, AbsRel +0.0035      warp 2.0  TCE 0.0304, AbsRel +0.0084
#   edge 2.0   AbsRel -0.0028, d1 +0.0025, synthetic RMSE 15.53 -> 14.51
#   d_max 600  no effect at 8k;  bins 128  worse on both domains -> dropped
# So warp is a monotone accuracy-for-stability trade and edge is free accuracy.
# This round finds the warp dose and tests the two together, then spends the
# 60k on whichever arm wins - the GPU is otherwise idle, so pre-committing the
# long run costs nothing but a discarded checkpoint if the pick is wrong.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

step() { echo; echo "################ $* ################"; date -Is; }

arm() {
    local name=$1; shift
    step "T1b $name"
    $PY scripts/train.py --config configs/main_v8.toml \
        --resume work_dirs/main_v8/latest.pt --resume-partial \
        --steps 8000 --seed 0 --work-dir "work_dirs/$name" "$@" || return
    $PY scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        "${REAL[@]}" "${RHOLD[@]}" --max-clips 100 --scores-tag real
    $PY scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        "${SYNTH[@]}" "${SHOLD[@]}" --max-clips 100 --scores-tag synth
}

arm t1-warp1.0             --warp-weight 1.0
arm t1-edge2-warp1         --warp-weight 1.0 --edge-weight 2.0
arm t1-edge2-warp2         --warp-weight 2.0 --edge-weight 2.0

# ---- T1-8: 60k on the winner ----------------------------------------------
# Accuracy is the constraint and stability the objective: among arms whose real
# AbsRel is no worse than the control's 0.1773 + one seed sigma (0.005), take
# the lowest real TCE. Round 1's arms are candidates too - t1-edge2.0 already
# beat the control on accuracy outright.
read -r BEST FLAGS <<<"$($PY - <<'PYEOF'
import pathlib, re
cand = {"t0-seed0-bin0.2": "", "t1-edge2.0": "--edge-weight 2.0",
        "t1-warp0.5": "--warp-weight 0.5", "t1-warp2.0": "--warp-weight 2.0",
        "t1-warp1.0": "--warp-weight 1.0",
        "t1-edge2-warp1": "--warp-weight 1.0 --edge-weight 2.0",
        "t1-edge2-warp2": "--warp-weight 2.0 --edge-weight 2.0"}
best = ("t0-seed0-bin0.2", "", 1e9)
for name, flags in cand.items():
    p = pathlib.Path("work_dirs") / name / "eval.txt"
    if not p.exists():
        continue
    # the real-domain eval is the one whose header lists tum+bonn; take the
    # LAST such block's MEAN(src) row (evals append)
    block = None
    for chunk in p.read_text().split("ckpt=")[1:]:
        if "tum_static" in chunk.split("\n")[0]:
            block = chunk
    if block is None:
        continue
    row = [l for l in block.splitlines() if "MEAN(src)" in l][-1].split()
    absrel, tce = float(row[3]), float(row[8])
    if absrel <= 0.1773 + 0.005 and tce < best[2]:
        best = (name, flags, tce)
print(best[0], best[1])
PYEOF
)"
step "T1-8 60k from the winning arm: $BEST  [$FLAGS]"
$PY scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/main_v8/latest.pt --resume-partial \
    --steps 60000 --seed 0 --work-dir work_dirs/v9-60k $FLAGS
$PY scripts/eval.py --ckpt work_dirs/v9-60k/latest.pt \
    "${REAL[@]}" "${RHOLD[@]}" --max-clips 100 --scores-tag real
$PY scripts/eval.py --ckpt work_dirs/v9-60k/latest.pt \
    "${SYNTH[@]}" "${SHOLD[@]}" --max-clips 100 --scores-tag synth
step "TIER 1b QUEUE DONE"
