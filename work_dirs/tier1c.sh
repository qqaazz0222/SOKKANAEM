#!/usr/bin/env bash
# The accuracy-first counterpart to work_dirs/v9-60k, queued behind it.
#   setsid nohup bash work_dirs/tier1c.sh > work_dirs/tier1c-console.log 2>&1 < /dev/null &
#
# tier1b's selector optimised real TCE subject to accuracy staying within one
# seed sigma, and picked edge2+warp2. But at 8k every arm carrying warp >= 1.0
# lands at real AbsRel 0.181-0.183 while edge 2.0 alone lands at 0.1745 - the
# two terms are not additive, warp dominates the trade. Accuracy is the axis we
# are already behind DA3 on (REPORT 4.20c) and PLAN's W2 framing explicitly
# does not chase an OPW/TCE win, so the final pick needs both candidates
# measured at 60k, not at 8k.
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

while pgrep -f 'work_dirs/tier1b.sh' > /dev/null; do sleep 600; done

echo; echo "################ T1-8b 60k, edge 2.0 only ################"; date -Is
$PY scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/main_v8/latest.pt --resume-partial \
    --steps 60000 --seed 0 --edge-weight 2.0 --work-dir work_dirs/v9-edge-60k
$PY scripts/eval.py --ckpt work_dirs/v9-edge-60k/latest.pt \
    "${REAL[@]}" "${RHOLD[@]}" --max-clips 100 --scores-tag real
$PY scripts/eval.py --ckpt work_dirs/v9-edge-60k/latest.pt \
    "${SYNTH[@]}" "${SHOLD[@]}" --max-clips 100 --scores-tag synth

# 32-frame clips too: T0-4 showed scale drift grows with clip length, and warp
# vs edge is exactly the kind of change that should move it.
for CK in v9-60k v9-edge-60k; do
    echo; echo "################ 32-frame drift — $CK ################"; date -Is
    $PY scripts/eval.py --ckpt "work_dirs/$CK/latest.pt" \
        "${REAL[@]}" "${RHOLD[@]}" --clip-len 32 --max-clips 30 --scores-tag drift32-real
done
echo "TIER 1c QUEUE DONE"
