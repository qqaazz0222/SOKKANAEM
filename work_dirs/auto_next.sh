#!/usr/bin/env bash
# Wait out the two running queues, then let the rule pick the branch.
set -u
until grep -q "queue17 done" work_dirs/queue17.log 2>/dev/null; do sleep 300; done
/home/hyunsu/miniforge3/envs/sokkanaem/bin/python scripts/decide_next.py m0-distil-s0 --launch
until grep -q "queue18 done" work_dirs/queue18.log 2>/dev/null; do sleep 300; done
echo "[$(date -Is)] auto_next done (queue18 finished too)"
