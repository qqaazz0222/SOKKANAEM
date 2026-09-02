#!/usr/bin/env bash
# C: the baselines' Bonn numbers broken down by region, so "Bonn 0.106 vs
# 0.058" can be attributed rather than just restated. The reference dumps were
# written without --regions.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
M=manifests/acc_real_L8.json
$PY scripts/eval_acc.py --manifest $M --regions --sharp --space disparity \
  --model hf:depth-anything/Depth-Anything-V2-Small-hf --tag da2-small-L8-regions
$PY scripts/eval_acc.py --manifest $M --regions --sharp --space disparity \
  --model hf:Intel/dpt-large --tag dpt-large-L8-regions
echo "[$(date -Is)] bonn refs done"
