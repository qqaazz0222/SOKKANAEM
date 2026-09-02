#!/usr/bin/env bash
# A -- 384px fine-tune (PLAN Stage D), from an already-sharpened checkpoint.
#
# The last untouched sharpness lever: DA2-small is scored at its official 518px
# and we train and infer at 256, so a 16px patch covers 1.5x more of the scene
# than its 14px one does. Scored twice on purpose: at 384 (what the model now
# is) and at 256 (what every gate reference was measured at) -- a sharpness
# number is not comparable across scoring resolutions, since the boundary band
# is defined on the resized GT.
set -eu
PY=${PY:-/home/hyunsu/miniforge3/envs/sokkanaem/bin/python}
BASE=${BASE:-work_dirs/d1-boundary3-s0/latest.pt}
STEPS=${STEPS:-6000}
name=${1:-a1-384-s0}; shift || true

$PY scripts/train.py --config configs/main_v8.toml \
  --resume "$BASE" --resume-partial \
  --size 384 --batch 2 --steps "$STEPS" --lr 5e-5 --warmup 500 \
  --full-res --boundary-weight 3.0 \
  --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
  --workers 16 --work-dir "work_dirs/$name" "$@"

for s in 384 256; do
  echo "[$(date -Is)] --- sharpness at ${s}px: $name ---"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" --size $s \
    --label "$name @${s}" --out outputs/sharpness/suite.jsonl
done
echo "[$(date -Is)] --- P1+P2 sealed L8: $name ---"
$PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
  --model "ours:work_dirs/$name/latest.pt" --regions --sharp --tag "$name-L8"
$PY scripts/acc_gate.py "work_dirs/acc/$name-L8.json" \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json \
  --sharp-label "$name @256" || true
