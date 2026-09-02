#!/usr/bin/env bash
# Q2 -- distil Q0 into the native 4.19M model.
#
# The two earlier teacher arms were rejected for reasons Q0 removes: raw DA2 is
# a RELATIVE model whose TUM scale collapses under a fit (15 clips of
# 0-crossing), so matching it had to go through an affine-invariant loss that
# dragged far-range structure. Q0 is metric and stable -- zero alignment
# failures under the median gauge -- so it can be matched in depth space, and
# it is dense, which is exactly the sharp INDOOR supervision the synthetic-only
# split (S1) could not provide and our Kinect GT does not contain.
#
# Two arms: teacher on top of the current best sharpening recipe, and teacher
# alone, so the two supervisions can be told apart.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
T=work_dirs/q0-calib-s0/latest.pt

bash scripts/finetune_arm.sh q2-distil-s0 --full-res --boundary-weight 3.0 \
  --q-teacher "$T" --q-weight 0.5 --q-grad-weight 0.5
bash scripts/finetune_arm.sh q2-teacheronly-s0 --full-res \
  --q-teacher "$T" --q-weight 1.0 --q-grad-weight 1.0

echo "=== Q2 table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
echo "[$(date -Is)] queue16 done"
