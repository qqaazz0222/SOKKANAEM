#!/usr/bin/env bash
# Stage 6: P2's reference rows on the sealed manifest (the earlier attempt died
# on a formatting bug in eval_acc's sharpness block, now fixed), then re-gate
# every arm against them.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "dynamic arm done" work_dirs/queue4.log 2>/dev/null; do sleep 60; done
bash work_dirs/sharp_refs.sh
echo "=== screening-protocol sharpness table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
for a in d1-fullres-s0 d0-control-s0 e1-boundary-s0 e2-rank-s0 e3-dynamic-s0; do
  d=work_dirs/acc/$a-L8.json
  [ -f "$d" ] || continue
  echo "=== gate: $a ==="
  $PY scripts/acc_gate.py "$d" --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp "$d" \
    --sharp-label "$a" || true
done
echo "[$(date -Is)] queue5 done"
