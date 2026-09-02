#!/usr/bin/env bash
# Q1 -- unfreeze the shape branch at a low LR, with a retention loss against
# the frozen original.
#
# Q0 fixed what a calibration head can fix: metric scale (median-gauge TUM
# 0.6451 -> 0.1097, Bonn 1.1188 -> 0.0871, zero alignment failures). What it
# could not fix is inherited: 10 clips still fail the 2-DOF fit because DA2's
# own TUM shape does, and alignment cancels a calibration head exactly. Only
# the shape branch can move that, so it moves -- slowly, and held to DA2 by a
# scale-shift-invariant retention term so the sharpness this track exists for
# does not quietly drain away.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
$PY scripts/train_q.py --steps 4000 --real-only --unfreeze \
  --shape-lr 1e-5 --retention-weight 1.0 --boundary-weight 0.5 \
  --work-dir work_dirs/q1-unfrozen-s0
echo "[$(date -Is)] --- Q1 sharpness ---"
$PY scripts/sharp_metric.py --ckpt work_dirs/q1-unfrozen-s0/latest.pt \
  --label "q1-unfrozen-s0" --out outputs/sharpness/suite.jsonl
echo "[$(date -Is)] --- Q1 P1+P2 sealed L8 ---"
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/q1-unfrozen-s0/latest.pt --regions --sharp \
  --tag q1-unfrozen-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/q1-unfrozen-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json \
  --sharp-label "q1-unfrozen-s0" || true
echo "[$(date -Is)] queue14 done"
