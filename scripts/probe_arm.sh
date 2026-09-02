#!/usr/bin/env bash
# One hard-subset probe arm, then its fit scores. See work_dirs/probe_capacity.sh
# for what the protocol is and why (PLAN.md 8.1): 256 dynamic/near-heavy real
# clips, trained and scored on the same clips, dense single-frame regime. The
# sealed holdout is never touched.
#
#   scripts/probe_arm.sh probe-d1-s0 --full-res
#   STEPS=6000 scripts/probe_arm.sh probe-m1-s0 --dim 288 --depth 6 --d-state 24
set -eu
PY=${PY:-/home/hyunsu/miniforge3/envs/sokkanaem/bin/python}
D=${D:-/home/hyunsu/dataset_ssd}
STEPS=${STEPS:-3000}
HARD=(rgbd_dataset_freiburg3_sitting_static rgbd_bonn_crowd3
      rgbd_bonn_person_tracking/ rgbd_bonn_moving_obstructing_box
      rgbd_bonn_synchronous)
name=$1; shift

hold=(); for h in "${HARD[@]}"; do hold+=(--holdout "$h"); done
echo "[$(date -Is)] === probe $name : $* ==="
$PY scripts/train.py --config configs/main_v8.toml \
  --data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset" \
  "${hold[@]}" --val-split --train-clips 256 \
  --clip-len 4 --batch 4 --size 256 --steps "$STEPS" \
  --max-skip 0 --no-augment --warp-weight 0 \
  --edge-weight 2.0 --spread-weight 0.5 \
  --workers 16 --work-dir "work_dirs/$name" "$@"
echo "[$(date -Is)] --- fit scores $name ---"
$PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
  --data "tum:$D/tum_static" "bonn:$D/bonn/rgbd_bonn_dataset" \
  --holdout "${HARD[@]}" --tau 0 --max-clips 64 \
  --label "$name (fit)" --out outputs/sharpness/probe.jsonl
