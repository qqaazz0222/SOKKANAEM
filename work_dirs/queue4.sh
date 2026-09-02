#!/usr/bin/env bash
# Stage 5: the dynamic-pixel weighted arm (the 2.66x gap of REPORT 4.43).
set -u
until grep -q "stage A loss arms done" work_dirs/queue3.log 2>/dev/null; do sleep 60; done
bash scripts/finetune_arm.sh e3-dynamic-s0 --dynamic-weight 1.0
echo "[$(date -Is)] dynamic arm done"
