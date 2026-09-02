#!/usr/bin/env bash
# The M0 distillation question, asked fairly this time.
#
# The first attempt is not evidence: M0's dim differs from v11's, so it could
# not inherit those 60k steps and ran 12k FROM SCRATCH, while every native arm
# it was compared against was a 4k fine-tune on top of a 60k checkpoint. Its
# AbsRel 0.2681 measures the missing 48k steps, not the teacher.
#
# So: 60k steps, the same budget the reported checkpoint had. This is the last
# open question of the native track -- whether an efficient student can hold
# DA2-class shape when it is given the same training the 4.19M model got.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "queue18 done" work_dirs/queue18.log 2>/dev/null; do sleep 300; done

$PY scripts/train.py --config configs/main_v8.toml \
  --dim 256 --depth 6 --d-state 24 --full-res \
  --steps 60000 --lr 3e-4 --warmup 2000 \
  --boundary-weight 3.0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
  --q-teacher work_dirs/q0-calib-s0/latest.pt --q-weight 1.0 --q-grad-weight 1.0 \
  --workers 16 --work-dir work_dirs/m0-distil-60k-s0
$PY scripts/sharp_metric.py --ckpt work_dirs/m0-distil-60k-s0/latest.pt \
  --label m0-distil-60k-s0 --out outputs/sharpness/suite.jsonl
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model ours:work_dirs/m0-distil-60k-s0/latest.pt --regions --sharp \
  --tag m0-distil-60k-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/m0-distil-60k-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label m0-distil-60k-s0 || true
echo "[$(date -Is)] queue21 done"
