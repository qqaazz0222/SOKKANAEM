#!/usr/bin/env bash
# Baseline table for the new sharpness suite (PLAN.md 10.1): the reported
# checkpoint, the patch-16 GT ceiling, and the two comparison points the
# sharpness gate is written against.
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
OUT=outputs/sharpness/suite.jsonl
run() { echo "[$(date -Is)] === $1 ==="; shift; $PY scripts/sharp_metric.py --out "$OUT" "$@" 2>&1 | grep -v "^ *$"; }
run "ours (reported v11)"  --ckpt work_dirs/v11-longclip-spread-s0/latest.pt --label "ours (reported v11)"
run "oracle p16"           --oracle-patch 16 --label "oracle p16"
run "DA2-small"            --baseline depth-anything/Depth-Anything-V2-Small-hf
run "DPT-Large"            --baseline Intel/dpt-large
echo "[$(date -Is)] suite done"
