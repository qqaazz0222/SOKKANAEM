#!/usr/bin/env bash
# Round 3 -- the sharpness mechanisms of DPT / DA v1 / DA V2, ported.
#
# What rounds 1-2 settled: the boundary term is the only lever that moves
# sharpness, it scales with its weight (grad_ratio 0.431 -> 0.457 at w=1 ->
# 0.578 at w=3), and it buys that with ringing (overshoot 0.316 -> 0.361
# against a 0.226 gate). So round 3 asks two things: can the ringing be paid
# off, and can the ceiling be raised by fixing the SUPERVISION rather than the
# weight -- which is Depth Anything V2's answer (train shape on synthetic GT
# only; real sensor GT is smeared at exactly the boundaries we are trying to
# sharpen).
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "queue9 done" work_dirs/queue9.log 2>/dev/null; do sleep 120; done

# S3 first: it unblocks the arm that already has the best sharpness
bash scripts/finetune_arm.sh s3-overshoot-s0 --full-res \
  --boundary-weight 3.0 --overshoot-weight 1.0
# S1/S2: DA V2's supervision split, then its noisy-label trimming
bash scripts/finetune_arm.sh s1-synthshape-s0 --full-res \
  --boundary-weight 1.0 --shape-on-synthetic
bash scripts/finetune_arm.sh s2-trim-s0 --full-res \
  --boundary-weight 1.0 --shape-on-synthetic --trim-real-band
# S4: MiDaS/DPT's own main sharpness lever, never swept here
bash scripts/finetune_arm.sh s4-msgrad2-s0 --full-res \
  --boundary-weight 1.0 --msgrad-weight 2.0
# S5: DPT's normalized fusion, re-tested under a loss that asks for detail
bash scripts/finetune_arm.sh s5-fusenorm-s0 --full-res \
  --boundary-weight 1.0 --fuse-norm

echo "=== round 3 table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
for a in s3-overshoot-s0 s1-synthshape-s0 s2-trim-s0 s4-msgrad2-s0 s5-fusenorm-s0; do
  echo "=== gate: $a ==="
  $PY scripts/acc_gate.py "work_dirs/acc/$a-L8.json" \
    --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp "work_dirs/acc/$a-L8.json" \
    --sharp-label "$a" || true
done
echo "[$(date -Is)] queue10 done"
