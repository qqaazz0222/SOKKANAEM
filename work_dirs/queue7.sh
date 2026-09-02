#!/usr/bin/env bash
# Everything after the D1/D0 pair, in the order the D1 result argues for.
#
# D1 alone moved nothing (grad_ratio 0.4262 against the reported 0.4323) and the
# checkpoint says why: `up_res`, the pixel-shuffle path that is the only way to
# put an edge inside a 2x2 block, stayed at its zero init (norm 0.008) while the
# full-res RGB path grew to 0.156. The decoder was given somewhere to put
# sharpness and the loss never asked for any. So the combined arm goes first.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "D1 arms done" work_dirs/d1-arms.log 2>/dev/null; do sleep 60; done

bash scripts/finetune_arm.sh d1-boundary-s0 --full-res --boundary-weight 1.0
bash scripts/finetune_arm.sh e1-boundary-s0 --boundary-weight 1.0
bash scripts/finetune_arm.sh e3-dynamic-s0  --dynamic-weight 1.0
bash scripts/finetune_arm.sh e2-rank-s0     --rank-weight 0.5

# P2's reference rows on the SEALED clips, so the gate stops comparing a
# subject measured on one clip set against references measured on another
bash work_dirs/sharp_refs.sh
echo "=== screening-protocol sharpness table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
echo "=== capacity probe table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/probe.jsonl
for a in d1-fullres-s0 d0-control-s0 d1-boundary-s0 e1-boundary-s0 \
         e3-dynamic-s0 e2-rank-s0; do
  d=work_dirs/acc/$a-L8.json
  [ -f "$d" ] || continue
  echo "=== gate: $a ==="
  $PY scripts/acc_gate.py "$d" --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp "$d" \
    --sharp-label "$a" || true
done
echo "[$(date -Is)] queue7 done"
