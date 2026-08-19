#!/usr/bin/env bash
# T5-25: how much of the drift in REPORT 4.30 is fixable by scheduling alone?
#
# The sawtooth says the keyframe refresh already repairs the damage; the
# question is what it costs to apply it more often. 32-frame clips, because an
# 8-frame clip cannot show a 30-frame period.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
for K in 5 10 15 30 60; do
    echo; echo "################ keyframe_every=$K ################"; date -Is
    $PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 32 --max-clips 30 --keyframe-every $K --scores-tag "kf$K"
done
echo "T5-25 DONE"; date -Is
