#!/usr/bin/env bash
# (2) A 10M-class student for the Q0 teacher.
#
# The 4.19M student could not absorb Q0 (accuracy 0.1234 -> 0.1354, sharpness
# below the boundary-loss arm on its own). The capacity probe that rejected M0
# measured a 256-clip FIT, where 10.8M simply memorized; with a dense teacher
# on 200k clips there is something for the extra capacity to fit. That is a
# different question from the one the probe answered, so it gets its own arm.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
T=work_dirs/q0-calib-s0/latest.pt
D=/home/hyunsu/dataset_ssd

$PY scripts/train.py --config configs/main_v8.toml \
  --dim 256 --depth 6 --d-state 24 --full-res \
  --steps 12000 --lr 3e-4 --warmup 1000 \
  --boundary-weight 3.0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
  --q-teacher "$T" --q-weight 1.0 --q-grad-weight 1.0 \
  --workers 16 --work-dir work_dirs/m0-distil-s0

$PY scripts/sharp_metric.py --ckpt work_dirs/m0-distil-s0/latest.pt \
  --label "m0-distil-s0" --out outputs/sharpness/suite.jsonl
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/m0-distil-s0/latest.pt --regions --sharp --tag m0-distil-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/m0-distil-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label "m0-distil-s0" || true
echo "[$(date -Is)] queue17 done"
