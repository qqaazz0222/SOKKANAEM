#!/usr/bin/env bash
# The 10.8M student, third attempt, with the recipe the diagnostics fixed.
#
# Attempt 1 (12k) was undertrained and uninformative. Attempt 2 (60k) minimised
# its loss into albedo: grad_ratio 1.80, overshoot 0.76, delta1 0.5427, with
# D1's RGB branch grown to norm 4.84. Two 8k probes then separated the causes:
# with the overshoot penalty on and the teacher's GRADIENT term off, the branch
# stays at 2.10 and the ratio at 0.56 -- and removing D1 entirely drops the
# ratio to 0.2023 and boundary F1 to 0.1183 at identical accuracy, so D1 is
# carrying the sharpness rather than leaking it.
#
# So: teacher in depth space only, boundary at 1, overshoot at 1, D1 on, 60k.
# Same watchdog as before -- a crash exits and gets resumed, a hang writes no
# log line for 15 minutes and gets killed into the same path.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
W=work_dirs/m0-fixed-60k-s0
STALL=${STALL:-900}
ARGS=(--config configs/main_v8.toml --dim 256 --depth 6 --d-state 24 --full-res
      --steps 60000 --lr 3e-4 --warmup 2000
      --boundary-weight 1.0 --overshoot-weight 1.0
      --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5
      --q-teacher work_dirs/q0-calib-s0/latest.pt --q-weight 1.0 --q-grad-weight 0.0
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
      pkill -9 -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null; sleep 10
    fi
  done
  wait $pid && break
  echo "[$(date -Is)] attempt $attempt ended, retrying in 30s"
  sleep 30
done

$PY scripts/sharp_metric.py --ckpt "$W/latest.pt" \
  --label m0-fixed-60k-s0 --out outputs/sharpness/suite.jsonl
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model "ours:$W/latest.pt" --regions --sharp --tag m0-fixed-60k-s0-L8
$PY scripts/acc_gate.py work_dirs/acc/m0-fixed-60k-s0-L8.json \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label m0-fixed-60k-s0 || true
$PY scripts/decide_next.py m0-fixed-60k-s0 || true
echo "[$(date -Is)] queue24 done"
