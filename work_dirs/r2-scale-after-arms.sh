#!/usr/bin/env bash
# Runs work_dirs/r2-scale.sh (r2 Major 1: baseline s_t statistics) once the
# sharpness-arms suite has released the GPU. Waits on the suite's shell, not on
# any single train.py, because the suite serialises arm C's training, its eval
# and its sharpness probe.
set -uo pipefail
cd "$(dirname "$0")/.."
ARMS_PID=${1:?usage: r2-scale-after-arms.sh <sharpness_arms.sh pid>}
while kill -0 "$ARMS_PID" 2>/dev/null; do sleep 120; done
echo "[$(date -Is)] arms suite (pid $ARMS_PID) gone"
# nothing else should be on the GPU either -- a stray eval would skew nothing
# but would slow both runs down
while pgrep -f "python.*scripts/(train|eval)" > /dev/null; do
    echo "[$(date -Is)] waiting for the GPU..."; sleep 120
done
exec work_dirs/r2-scale.sh
