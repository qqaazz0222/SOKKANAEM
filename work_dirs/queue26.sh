#!/usr/bin/env bash
# Two short arms, chosen because each has evidence behind it and costs an hour.
#
# Every from-scratch 10.8M run oversharpened (four attempts, 30 GPU-hours each,
# grad_ratio 1.80 and 2.25); every FINE-TUNE from the reported checkpoint stayed
# sane. So the fixed recipe -- boundary 1.0, overshoot 1.0, Q0 as a depth-space
# teacher, no teacher gradient term -- goes where it has always worked, on top
# of the 4.19M checkpoint that already has 60k steps of shape behind it.
#
# And the Q0 adapter gets the one setting not yet tried: retention 4 held the
# shape (grad_ratio 0.409 -> 0.645) while gt-weight 0.2 let the metric scale
# drift (AbsRel 0.1998). Both terms at full strength is the middle of that
# trade, and it is the last piece the streaming candidate needs.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
T=work_dirs/q0-calib-s0/latest.pt

bash scripts/finetune_arm.sh v11-teacher-s0 --full-res \
  --boundary-weight 1.0 --overshoot-weight 1.0 \
  --q-teacher "$T" --q-weight 1.0 --q-grad-weight 0.0

$PY scripts/train_q.py --steps 4000 --clip-len 8 --batch 1 --real-only \
  --temporal --warp-weight 2.0 --retention-weight 4.0 --gt-weight 1.0 \
  --work-dir work_dirs/qt-balanced-s0
$PY scripts/sharp_metric.py --ckpt work_dirs/qt-balanced-s0/latest.pt \
  --label qt-balanced-s0 --out outputs/sharpness/suite.jsonl
for L in 8 256; do
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L$L.json \
    --model ours:work_dirs/qt-balanced-s0/latest.pt --per-frame \
    --tag "qt-balanced-s0-L$L-pf"
done
$PY scripts/acc_gate.py work_dirs/acc/qt-balanced-s0-L8-pf.json \
  work_dirs/acc/qt-balanced-s0-L256-pf.json --gauge scaleshift/frame \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label qt-balanced-s0 || true
echo "[$(date -Is)] queue26 done"
