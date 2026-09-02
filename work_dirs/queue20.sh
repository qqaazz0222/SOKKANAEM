#!/usr/bin/env bash
# Branch B -- the teacher moves sharpness but does not close the gap. DA V2's
# own answer to that was more pseudo-labelled data, not a bigger student, so
# widen what the teacher is asked about: every source, longer, and with the
# metric term on the teacher rather than the sensor wherever the sensor is
# silent (that is already how the teacher loss is masked).
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
$PY scripts/train.py --config configs/main_v8.toml \
  --dim 256 --depth 6 --d-state 24 --full-res \
  --steps 30000 --lr 3e-4 --warmup 1000 \
  --boundary-weight 3.0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
  --q-teacher work_dirs/q0-calib-s0/latest.pt --q-weight 2.0 --q-grad-weight 2.0 \
  --workers 16 --work-dir work_dirs/m0-distil-long-s0
$PY scripts/sharp_metric.py --ckpt work_dirs/m0-distil-long-s0/latest.pt \
  --label m0-distil-long-s0 --out outputs/sharpness/suite.jsonl
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/m0-distil-long-s0/latest.pt --regions --sharp \
  --tag m0-distil-long-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/m0-distil-long-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label m0-distil-long-s0 || true
echo "[$(date -Is)] queue20 done"
