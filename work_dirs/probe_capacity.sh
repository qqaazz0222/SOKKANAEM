#!/usr/bin/env bash
# PLAN.md 8.1 -- hard-subset saturation probe: does capacity (dim/depth/d_state)
# move the measured bottleneck (dynamic + near-range error), or is the model
# already at its data/decoder limit?
#
# Deliberately a FIT test, not a generalization test: both arms train on the
# same 256 dynamic/near-heavy real clips and are scored on them. A bigger model
# that cannot fit the hard clips any better has no capacity headroom to spend,
# and the 60k run is then not worth its two days. The sealed holdout
# (walking_static, crowd2, person_tracking2, static_close_far) is untouched by
# the --holdout list below.
#
# Dense single-frame regime (PLAN Stage A): --max-skip 0 so every patch is
# active, no augmentation, and no warp term (a temporal-consistency loss has
# nothing to say about per-frame shape). The scoring pass needs --tau 0 for the
# same reason: scored through the change detector instead, a dense-trained arm
# is read off-distribution, and the first run of this probe had the two arms
# off-distribution by different amounts.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
HARD=(--holdout rgbd_dataset_freiburg3_sitting_static
      --holdout rgbd_bonn_crowd3
      --holdout rgbd_bonn_person_tracking/
      --holdout rgbd_bonn_moving_obstructing_box
      --holdout rgbd_bonn_synchronous)
STEPS=${STEPS:-3000}

arm () {   # arm <name> <dim> <depth> <d_state> [extra...]
  local name=$1 dim=$2 depth=$3 ds=$4; shift 4
  echo "[$(date -Is)] === probe $name (dim $dim depth $depth d_state $ds) ==="
  $PY scripts/train.py --config configs/main_v8.toml \
    --data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset" \
    "${HARD[@]}" --val-split --train-clips 256 \
    --clip-len 4 --batch 4 --size 256 --steps "$STEPS" \
    --max-skip 0 --no-augment --warp-weight 0 \
    --edge-weight 2.0 --spread-weight 0.5 \
    --dim "$dim" --depth "$depth" --d-state "$ds" "$@" \
    --workers 16 --work-dir "work_dirs/$name"
  echo "[$(date -Is)] --- fit scores $name ---"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
    --data "tum:$D/tum_static" "bonn:$D/bonn/rgbd_bonn_dataset" \
    --holdout rgbd_dataset_freiburg3_sitting_static rgbd_bonn_crowd3 \
              rgbd_bonn_person_tracking/ rgbd_bonn_moving_obstructing_box \
              rgbd_bonn_synchronous \
    --tau 0 --max-clips 64 --label "$name (fit)" --out outputs/sharpness/probe.jsonl
}

arm probe-cur-s0 192 4 16
arm probe-m0-s0  256 6 24
echo "[$(date -Is)] capacity probe done"
