#!/usr/bin/env bash
# The 60k distillation with BOTH failure modes covered.
#
# Attempt 1 died at step 8500 with a SIGSEGV in a worker on a truncated image;
# the resume loop handled that. Attempt 2 did something the loop cannot see: it
# HUNG at step 28300 -- process alive, one core spinning, no log line for eight
# hours and forty minutes. A crash exits and gets retried; a deadlock does not,
# so it has to be detected from the outside. This runner watches the training
# log's mtime and kills a run that has written nothing for 15 minutes, which
# then falls through to the same resume path.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
W=work_dirs/m0-distil-60k-s0
STALL=${STALL:-900}
ARGS=(--config configs/main_v8.toml --dim 256 --depth 6 --d-state 24 --full-res
      --steps 60000 --lr 3e-4 --warmup 2000
      --boundary-weight 3.0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5
      --q-teacher work_dirs/q0-calib-s0/latest.pt --q-weight 1.0 --q-grad-weight 1.0
      --workers 8 --work-dir "$W")

for attempt in $(seq 1 60); do
  echo "[$(date -Is)] attempt $attempt"
  if [ -f "$W/latest.pt" ]; then
    $PY scripts/train.py "${ARGS[@]}" --resume "$W/latest.pt" &
  else
    $PY scripts/train.py "${ARGS[@]}" &
  fi
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "$W/train.log" 2>/dev/null || date +%s) ))
    if [ "$age" -gt "$STALL" ]; then
      echo "[$(date -Is)] no log line for ${age}s -- killing $pid and resuming"
      pkill -9 -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null
      sleep 10
    fi
  done
  wait $pid && break
  echo "[$(date -Is)] attempt $attempt ended, retrying in 30s"
  sleep 30
done

$PY scripts/sharp_metric.py --ckpt "$W/latest.pt" \
  --label m0-distil-60k-s0 --out outputs/sharpness/suite.jsonl
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model "ours:$W/latest.pt" --regions --sharp --tag m0-distil-60k-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/m0-distil-60k-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label m0-distil-60k-s0 || true
echo "[$(date -Is)] queue21 done"
