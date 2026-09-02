#!/usr/bin/env bash
# Stage 7: re-score probe-cur-s0's fit with the dynamic-region column, which was
# added after that arm ran, so both probe arms are read from the same table.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
until grep -q "queue5 done" work_dirs/queue5.log 2>/dev/null; do sleep 60; done
$PY scripts/sharp_metric.py --ckpt work_dirs/probe-cur-s0/latest.pt \
  --data "tum:$D/tum_static" "bonn:$D/bonn/rgbd_bonn_dataset" \
  --holdout rgbd_dataset_freiburg3_sitting_static rgbd_bonn_crowd3 \
            rgbd_bonn_person_tracking/ rgbd_bonn_moving_obstructing_box \
            rgbd_bonn_synchronous \
  --max-clips 64 --label "probe-cur-s0 (fit)" --out outputs/sharpness/probe.jsonl
echo "=== capacity probe table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/probe.jsonl
echo "[$(date -Is)] queue6 done"
