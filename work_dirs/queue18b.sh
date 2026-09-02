#!/usr/bin/env bash
# The evaluation half of queue18, rerun after the batch-chunking fix. Training
# itself completed (work_dirs/qt-stream-s0/latest.pt, 4000 steps).
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
$PY scripts/sharp_metric.py --ckpt work_dirs/qt-stream-s0/latest.pt \
  --label "qt-stream-s0" --out outputs/sharpness/suite.jsonl
for L in 8 256; do
  for m in qt-stream-s0 q0-calib-s0; do
    $PY scripts/eval_acc.py --manifest manifests/acc_real_L$L.json \
      --model "ours:work_dirs/$m/latest.pt" --per-frame --tag "$m-L$L-pf"
  done
done
echo "=== P3, per-frame gauge ==="
$PY scripts/acc_gate.py work_dirs/acc/qt-stream-s0-L8-pf.json \
  work_dirs/acc/qt-stream-s0-L256-pf.json --gauge scaleshift/frame \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label "qt-stream-s0" || true
echo "=== P3 control: frozen, stateless Q0 ==="
$PY scripts/acc_gate.py work_dirs/acc/q0-calib-s0-L8-pf.json \
  work_dirs/acc/q0-calib-s0-L256-pf.json --gauge scaleshift/frame || true
echo "[$(date -Is)] queue18 done"
