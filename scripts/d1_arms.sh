#!/usr/bin/env bash
# PLAN.md 8.3 -- does learned full-resolution upsampling move sharpness?
#
# The D1 residual is zero-init, so a fine-tune from the reported checkpoint
# starts at exactly its output: the arm can only be measured as a delta on top
# of v11, which is the cheapest honest way to ask the question (a from-scratch
# 60k pair costs four days to answer the same thing). D0 runs the identical
# fine-tune WITHOUT the residual, because otherwise the comparison measures
# "4k more steps" rather than the decoder change.
#
# Scored on the sealed real holdout: the sharpness suite (P2) and eval_acc's
# L8 manifest (P1), with region breakdown for the dynamic/near bottleneck.
set -eu
PY=${PY:-/home/hyunsu/miniforge3/envs/sokkanaem/bin/python}
BASE=${BASE:-work_dirs/v11-longclip-spread-s0/latest.pt}
STEPS=${STEPS:-4000}

arm () {   # arm <name> [extra train flags...]
  local name=$1; shift
  echo "[$(date -Is)] === $name : $* ==="
  $PY scripts/train.py --config configs/main_v8.toml \
    --resume "$BASE" --resume-partial \
    --steps "$STEPS" --lr 1e-4 --warmup 500 \
    --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
    --workers 16 --work-dir "work_dirs/$name" "$@"

  echo "[$(date -Is)] --- P2 sharpness, sealed holdout: $name ---"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
    --label "$name" --out outputs/sharpness/suite.jsonl
  echo "[$(date -Is)] --- P1+P2 on the sealed L8 manifest: $name ---"
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
    --model "ours:work_dirs/$name/latest.pt" --regions --sharp --tag "$name-L8"
  # the dump's own sharpness is on the sealed clips; the DA2/reported reference
  # rows still come from the screening JSONL until those two are re-measured
  # there as well (see work_dirs/sharp_refs.sh)
  $PY scripts/acc_gate.py "work_dirs/acc/$name-L8.json" \
    --sharp outputs/sharpness/suite.jsonl \
    --sharp "work_dirs/acc/$name-L8.json" --sharp-label "$name" || true
}

arm d1-fullres-s0 --full-res
arm d0-control-s0
echo "[$(date -Is)] D1 arms done"
