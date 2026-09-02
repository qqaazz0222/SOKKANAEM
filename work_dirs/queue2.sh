#!/usr/bin/env bash
# Stage 3 of the serial GPU queue: sealed sharpness references, then the two
# Stage A loss arms (PLAN §6, fixed weights, one variable each).
set -u
while pgrep -f "d1_arms.sh" > /dev/null; do sleep 60; done
bash work_dirs/sharp_refs.sh
bash scripts/finetune_arm.sh e1-boundary-s0 --boundary-weight 1.0
bash scripts/finetune_arm.sh e2-rank-s0 --rank-weight 0.5
echo "[$(date -Is)] queue2 done"
