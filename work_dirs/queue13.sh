#!/usr/bin/env bash
# Q0 -- PLAN §7's floor. Frozen DA2 shape, trained metric calibration.
#
# Worth stating what this can and cannot answer: P1 aligns with a 2-DOF
# scale+shift fit, which cancels a calibration head, so Q0's ALIGNED accuracy
# is DA2's by construction -- Bonn 0.0578 (a pass) and TUM 0.3464 (a fail that
# is really 15 clips of 0-crossing, REPORT §4.43). The interesting part is
# exactly that failure: those crossings come from fitting a RELATIVE model's
# disparity, and Q0 emits metric depth, so its fit should sit near identity and
# not invert. If that holds, this track clears the accuracy gate on both
# sources with DA2's sharpness intact -- which is the whole point of §7.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
$PY scripts/train_q.py --steps 4000 --real-only --work-dir work_dirs/q0-calib-s0
echo "[$(date -Is)] --- Q0 sharpness (screening protocol) ---"
$PY scripts/sharp_metric.py --ckpt work_dirs/q0-calib-s0/latest.pt \
  --label "q0-calib-s0" --out outputs/sharpness/suite.jsonl
echo "[$(date -Is)] --- Q0 P1+P2 sealed L8 ---"
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/q0-calib-s0/latest.pt --regions --sharp --tag q0-calib-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/q0-calib-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json \
  --sharp-label "q0-calib-s0" || true
echo "[$(date -Is)] queue13 done"
