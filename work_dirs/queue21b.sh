#!/usr/bin/env bash
# The 60k distillation, restarted with crash recovery.
#
# The first attempt died at step 8500 with a SIGSEGV (core dumped) right after
# "skipping unreadable clip ... image file is truncated" -- a native-level
# crash in a worker on this scraped dataset, not a Python exception the loader
# can catch. Ten hours of GPU time then sat idle. train.py checkpoints model,
# optimizer and step together, so the cheap answer is to resume rather than to
# hunt a decoder bug: each restart picks up from the last save.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
W=work_dirs/m0-distil-60k-s0
ARGS=(--config configs/main_v8.toml --dim 256 --depth 6 --d-state 24 --full-res
      --steps 60000 --lr 3e-4 --warmup 2000
      --boundary-weight 3.0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5
      --q-teacher work_dirs/q0-calib-s0/latest.pt --q-weight 1.0 --q-grad-weight 1.0
      --workers 16 --work-dir "$W")

for attempt in $(seq 1 40); do
  if [ -f "$W/latest.pt" ]; then
    echo "[$(date -Is)] attempt $attempt: resuming from $W/latest.pt"
    $PY scripts/train.py "${ARGS[@]}" --resume "$W/latest.pt" && break
  else
    echo "[$(date -Is)] attempt $attempt: fresh start"
    $PY scripts/train.py "${ARGS[@]}" && break
  fi
  echo "[$(date -Is)] attempt $attempt died (exit $?), retrying in 30s"
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
