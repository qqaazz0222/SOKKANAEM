#!/usr/bin/env bash
# Continues the warp_weight=0.0 ablation after the previous run got killed
# (twice) with a crash-insurance checkpoint at step 2000/8000. This resumes
# full state (optimizer + step counter, NOT --resume-partial, which would
# restart the schedule from step 0) and finishes the same 8000-step budget,
# then runs the eval + range_probe steps the original r3-warp0.0.sh never
# reached.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
WD=work_dirs/v11-warp0.0

echo "======== resume: warp_weight=0.0 from step 2000 ========"; date -Is
$PY scripts/train.py --config configs/main_v8.toml \
    --resume "$WD/latest.pt" \
    --clip-len 24 --batch 2 --steps 8000 --seed 0 \
    --warp-weight 0.0 --edge-weight 2.0 --spread-weight 0.5 \
    --work-dir "$WD"
echo "train done"; date -Is

HOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
      --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
DATA=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")

for L in 8 32 256; do
    for K in 30 60; do
        echo; echo "---- eval L$L K$K ----"; date -Is
        $PY scripts/eval.py --ckpt "$WD/latest.pt" "${DATA[@]}" "${HOLD[@]}" \
            --clip-len "$L" --size 256 --max-clips 100 --align median \
            --gate-mode delta --keyframe-every "$K" --dense-above 0.4 \
            --bin-temp 1.0 --scores-tag "warp0.0-L${L}K${K}" \
            | tee -a "$WD/eval.txt"
    done
done

echo; echo "---- range probe ----"; date -Is
$PY scripts/range_probe.py --ckpt "$WD/latest.pt" | tee "$WD/range.txt"

echo; echo "R3 WARP0.0 DONE"; date -Is
