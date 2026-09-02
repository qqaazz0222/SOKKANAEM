#!/usr/bin/env bash
# P2's reference rows on the SEALED manifest, so the sharpness gate stops
# comparing a subject measured on one clip set against references measured on
# another (sharp_metric.py's 100-clips-per-source screening protocol).
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
M=manifests/acc_real_L8.json
$PY scripts/eval_acc.py --manifest $M --sharp --regions \
  --model ours:work_dirs/v11-longclip-spread-s0/latest.pt --tag ours-L8-sharp
$PY scripts/eval_acc.py --manifest $M --sharp --space disparity \
  --model hf:depth-anything/Depth-Anything-V2-Small-hf --tag da2-small-L8-sharp
$PY scripts/eval_acc.py --manifest $M --sharp --space disparity \
  --model hf:Intel/dpt-large --tag dpt-large-L8-sharp
