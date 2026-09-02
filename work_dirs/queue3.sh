#!/usr/bin/env bash
# Stage 4: the two Stage A loss arms, once every earlier GPU job is really
# done. Waits on the log lines each stage prints when it finishes -- a pgrep
# check alone fires early if the stage has not started yet, which is how the
# first attempt collided with the capacity probe.
set -u
done_line () { grep -q "$2" "$1" 2>/dev/null; }
until done_line work_dirs/probe-capacity.log "capacity probe done" \
   && done_line work_dirs/d1-arms.log "D1 arms done" \
   && ! pgrep -f "sharp_refs.sh" > /dev/null; do sleep 60; done
bash scripts/finetune_arm.sh e1-boundary-s0 --boundary-weight 1.0
bash scripts/finetune_arm.sh e2-rank-s0 --rank-weight 0.5
echo "[$(date -Is)] stage A loss arms done"
