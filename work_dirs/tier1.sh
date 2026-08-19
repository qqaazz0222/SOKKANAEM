#!/usr/bin/env bash
# PLAN.md Tier 1, one detached queue (setsid nohup) so it survives the session.
# Every arm is 8k steps from the same v8 init, then the same two evals — the
# T0-2 seed sweep says real AbsRel moves +-0.005 and synthetic delta1 +-0.015
# between seeds, so only differences bigger than that count.
#   setsid nohup bash work_dirs/tier1.sh > work_dirs/tier1-console.log 2>&1 < /dev/null &
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

# name, config, extra train flags...
arm() {
    local name=$1 cfg=$2; shift 2
    step "T1 $name"
    $PY scripts/train.py --config "configs/$cfg.toml" \
        --resume work_dirs/main_v8/latest.pt --resume-partial \
        --steps 8000 --seed 0 --work-dir "work_dirs/$name" "$@" || return
    $PY scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        "${REAL[@]}" "${RHOLD[@]}" --max-clips 100 --scores-tag real
    $PY scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        "${SYNTH[@]}" "${SHOLD[@]}" --max-clips 100 --scores-tag synth
}

# T1-5 · W6 far-range RMSE. bin_probe.py already ruled out bin COUNT below
# 80 m (quantization floor 0.0000 AbsRel there), so these two test the range.
arm t1-binrange t1_binrange
arm t1-bin128   t1_bin128

# T1-6 · W2 warp-residual loss (training-time TCE). 0.5 is the same order as
# msgrad_weight; the term is a log-space L1 like si_log.
arm t1-warp0.5 main_v8 --warp-weight 0.5
arm t1-warp2.0 main_v8 --warp-weight 2.0

# T1-7 · W7 foreground / depth-discontinuity weighting.
arm t1-edge0.5 main_v8 --edge-weight 0.5
arm t1-edge2.0 main_v8 --edge-weight 2.0

# seed-0 control on the untouched recipe: T0-2's seed0-bin0.2 is exactly that
# run, so no extra arm is needed — compare against work_dirs/t0-seed0-bin0.2.
step "TIER 1 QUEUE DONE"
