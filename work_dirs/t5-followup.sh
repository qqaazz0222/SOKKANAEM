#!/usr/bin/env bash
# After T5-24: did the sawtooth flatten, did the best keyframe period move, and
# does the range-compression term help? Chained so they never share the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v10-longclip/latest.pt
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

echo "################ A. frame-index probe on the long-clip model ################"; date -Is
CLIP_LEN=32 MAX_CLIPS=30 $PY scripts/frame_index_probe.py 2>&1 | sed 's/latest/v9/' > /dev/null || true
sed -i 's|work_dirs/v9-60k/latest.pt|work_dirs/v10-longclip/latest.pt|' scripts/frame_index_probe.py
CLIP_LEN=32 MAX_CLIPS=30 $PY scripts/frame_index_probe.py
sed -i 's|work_dirs/v10-longclip/latest.pt|work_dirs/v9-60k/latest.pt|' scripts/frame_index_probe.py

echo; echo "################ B. keyframe sweep, long-clip model ################"; date -Is
for K in 5 10 15 30 60; do
    echo; echo "---- keyframe_every=$K ----"
    $PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 32 --max-clips 30 --keyframe-every $K --scores-tag "v10kf$K"
done

echo; echo "################ C. T5-33 range-compression probe ################"; date -Is
for W in 0.0 0.5 2.0; do
    DIR=work_dirs/t5-spread$W
    echo; echo "---- spread_weight=$W ----"; date -Is
    $PY scripts/train.py --config configs/main_v8.toml \
        --resume work_dirs/v9-60k/latest.pt --resume-partial \
        --steps 8000 --seed 0 --warp-weight 2.0 --edge-weight 2.0 \
        --spread-weight $W --work-dir "$DIR"
    $PY scripts/eval.py --ckpt "$DIR/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
        --max-clips 100 --scores-tag real
    echo "-- range ratio --"
    $PY scripts/range_probe.py --ckpt "$DIR/latest.pt"
done
echo "FOLLOWUP DONE"; date -Is
