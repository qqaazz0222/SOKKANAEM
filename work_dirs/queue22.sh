#!/usr/bin/env bash
# The temporal adapter, retried as SELF-DISTILLATION.
#
# First attempt: zero-init adapter + our GT's si-log + warp for 4k steps. It
# reproduced Q0 exactly at step 0 and then walked away from it -- grad_ratio
# 0.709 -> 0.409, drift 22.2% -> 33.5%. The zero-init guarantee holds at step
# zero only, and the signal that moved it is the same sensor GT that broke Q1,
# Q2 and S1.
#
# So the adapter is now trained to PRESERVE Q0's own per-frame output
# (retention, weight 4) while the warp term asks for temporal consistency, and
# the GT term is turned down to 0.2 -- it exists only to keep the metric scale
# from wandering.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
$PY scripts/train_q.py --steps 4000 --clip-len 8 --batch 1 --real-only \
  --temporal --warp-weight 2.0 --retention-weight 4.0 --gt-weight 0.2 \
  --work-dir work_dirs/qt-retain-s0
$PY scripts/sharp_metric.py --ckpt work_dirs/qt-retain-s0/latest.pt \
  --label "qt-retain-s0" --out outputs/sharpness/suite.jsonl
for L in 8 256; do
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L$L.json \
    --model ours:work_dirs/qt-retain-s0/latest.pt --per-frame --tag "qt-retain-s0-L$L-pf"
done
echo "=== P3 ==="
$PY scripts/acc_gate.py work_dirs/acc/qt-retain-s0-L8-pf.json \
  work_dirs/acc/qt-retain-s0-L256-pf.json --gauge scaleshift/frame \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label "qt-retain-s0" || true
echo "[$(date -Is)] queue22 done"
