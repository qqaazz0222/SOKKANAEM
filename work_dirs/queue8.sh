#!/usr/bin/env bash
# Round 2. What the first round settled: the boundary term is the only one that
# moves sharpness (precision 0.556 -> 0.630, grad_ratio +0.026 in 4k steps), and
# D1's shuffle path only earns its keep when that term is on (D1+boundary beats
# boundary alone on every sharpness column). Round 2 asks whether the effect
# scales -- with the weight, and with the step budget -- and then combines the
# terms that were accuracy-positive.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python

bash scripts/finetune_arm.sh d1-boundary3-s0 --full-res --boundary-weight 3.0
STEPS=12000 bash scripts/finetune_arm.sh d1-boundary-long-s0 --full-res --boundary-weight 1.0
bash scripts/finetune_arm.sh combined-s0 --full-res --boundary-weight 1.0 \
  --dynamic-weight 1.0 --rank-weight 0.5

echo "=== round 2 table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
for a in d1-boundary3-s0 d1-boundary-long-s0 combined-s0; do
  echo "=== gate: $a ==="
  $PY scripts/acc_gate.py "work_dirs/acc/$a-L8.json" \
    --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp "work_dirs/acc/$a-L8.json" \
    --sharp-label "$a" || true
done
echo "[$(date -Is)] queue8 done"
