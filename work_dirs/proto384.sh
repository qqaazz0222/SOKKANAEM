#!/usr/bin/env bash
# The 384 arm looked better at 384 and worse at 256, and the sealed protocol
# fixes 256 -- so the question is not "does 384 help" but "does the ordering
# against the baselines change when everyone is scored at 384". Baselines are
# run at their own official input resolution either way (DA2 518), so scoring
# resolution is the only thing that moves here.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
O=outputs/sharpness/proto384.jsonl
$PY scripts/sharp_metric.py --size 384 --ckpt work_dirs/d1-boundary3-s0/latest.pt \
  --label "ours b3 (256-trained) @384" --out $O
$PY scripts/sharp_metric.py --size 384 --ckpt work_dirs/a1-384-s0/latest.pt \
  --label "ours a1 (384-trained) @384" --out $O
$PY scripts/sharp_metric.py --size 384 \
  --baseline depth-anything/Depth-Anything-V2-Small-hf --label "DA2-small @384" --out $O
$PY scripts/sharp_metric.py --size 384 --baseline Intel/dpt-large \
  --label "DPT-Large @384" --out $O
echo "=== 384-scoring protocol ==="
$PY scripts/sharp_metric.py --table $O
echo "[$(date -Is)] proto384 done"
