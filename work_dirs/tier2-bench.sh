#!/usr/bin/env bash
# PLAN.md T2-9 measurement. Waits for the Tier 1 training queue to release the
# GPU first — the last wall-clock round was contaminated by other processes and
# read 13 ms for a path that actually takes 47 ms (REPORT §4.19f).
#   setsid nohup bash work_dirs/tier2-bench.sh > work_dirs/tier2-bench.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CKPT=work_dirs/v8-teacherfree-60k/latest.pt
ACTIVE="0.05 0.10 0.22 0.32 0.50 0.70"

while pgrep -f 'work_dirs/tier1.sh' > /dev/null; do sleep 300; done
sleep 60                       # let the last eval's memory actually free
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

run() { echo; echo "################ $* ################"; date -Is
        $PY scripts/bench.py --ckpt $CKPT --active $ACTIVE --repeat 3 "$@"; }

run --cache off                         # dense, eager
run --cache off --compile               # dense, CUDA graphs (the bar to beat)
run --cache on                          # sparse today: a new shape every frame
run --cache on --bucket 64              # what the padding alone costs
run --cache on --bucket 64 --compile    # the point of T2-9
run --cache on --bucket 64 --compile --half
echo "T2-9 BENCH DONE"
