#!/usr/bin/env bash
# One fine-tune arm from the reported checkpoint, then P1+P2 on the sealed
# clips. Same protocol as scripts/d1_arms.sh (which predates this file and
# holds the D1/D0 pair): 4k steps at lr 1e-4 on the v11 recipe, so an arm is a
# delta on a known checkpoint rather than a fresh run with its own luck.
#
#   scripts/finetune_arm.sh e1-boundary-s0 --boundary-weight 1.0
set -eu
PY=${PY:-/home/hyunsu/miniforge3/envs/sokkanaem/bin/python}
BASE=${BASE:-work_dirs/v11-longclip-spread-s0/latest.pt}
STEPS=${STEPS:-4000}
name=$1; shift

echo "[$(date -Is)] === $name : $* ==="
$PY scripts/train.py --config configs/main_v8.toml \
  --resume "$BASE" --resume-partial \
  --steps "$STEPS" --lr 1e-4 --warmup 500 \
  --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
  --workers 16 --work-dir "work_dirs/$name" "$@"

echo "[$(date -Is)] --- P2 screening protocol: $name ---"
$PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
  --label "$name" --out outputs/sharpness/suite.jsonl
echo "[$(date -Is)] --- P1+P2 sealed L8 manifest: $name ---"
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model "ours:work_dirs/$name/latest.pt" --regions --sharp --tag "$name-L8"
# the sealed dump goes last so its numbers win over the screening protocol's
$PY scripts/acc_gate.py "work_dirs/acc/$name-L8.json" \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp "work_dirs/acc/$name-L8.json" --sharp-label "$name" || true
