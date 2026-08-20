#!/usr/bin/env bash
# The keyframe re-sweep (REPORT 4.34) is all 32-frame, but the comparison group
# in 4.29 was measured at 8. This gets the missing number so the adoption
# decision and the baseline table can be settled on one protocol.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)

while pgrep -f 't5-followup.sh' > /dev/null; do sleep 300; done
echo "followup finished, measuring v10 at keyframe 60 on the 8-frame protocol"; date -Is
for K in 30 60; do
    echo; echo "---- v10, keyframe $K, 8-frame real ----"
    $PY scripts/eval.py --ckpt work_dirs/v10-longclip/latest.pt "${REAL[@]}" \
        "${RHOLD[@]}" --max-clips 100 --keyframe-every $K --scores-tag "v10-8f-kf$K"
done
echo; echo "---- v10, keyframe 60, 8-frame synthetic ----"
$PY scripts/eval.py --ckpt work_dirs/v10-longclip/latest.pt "${SYNTH[@]}" \
    "${SHOLD[@]}" --max-clips 100 --keyframe-every 60 --scores-tag v10-8f-synth
echo "8FRAME CHECK DONE"; date -Is
