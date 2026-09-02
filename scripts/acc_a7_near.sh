#!/usr/bin/env bash
# PLAN_ACC A7: weight the loss towards the band where the error actually is.
#
# REPORT §4.43 (A6) measured our error against a 343M reference on the same
# pixels: 1.13x overall, 1.15x in the depth-boundary band, but 2.38x inside
# 2 m and 2.66x on independently moving pixels -- and 73% of those moving
# pixels are inside 2 m, in both sources. So a depth band is a 73%-effective
# motion mask that costs one comparison against GT.
#
# This is the SAME 8k stage that produced the reported checkpoint (v11 is
# v10-longclip + 8k with warp 2.0 / edge 2.0 / spread 0.5), with --near-weight
# added and nothing else changed. So it is a single-variable comparison against
# work_dirs/v11-longclip-spread-s0, not an 8k probe standing in for a 60k
# question -- the reported model is itself this stage.
#
# Two weights, because one arm cannot tell "no effect" from "wrong magnitude".
# Judgment is G1 AND the A6 region table together: moving the near-range
# number while breaking the rest is not a win.
set -eu
cd "$(dirname "$0")/.."

BASE="work_dirs/v10-longclip/latest.pt"
COMMON="--config configs/main_v8.toml --resume $BASE --resume-partial
        --clip-len 24 --batch 2 --steps 8000 --seed 0
        --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5"
# 8 each: the box has 32 cores and the two arms share them
W="${WORKERS:-8}"

for NW in ${WEIGHTS:-1.0 3.0}; do
  DIR="work_dirs/v13-near${NW}-s0"
  echo "=== $DIR"
  python scripts/train.py $COMMON --workers "$W" \
    --near-weight "$NW" --work-dir "$DIR" > "${DIR}.log" 2>&1 &
done
wait
echo "training done"

for NW in ${WEIGHTS:-1.0 3.0}; do
  DIR="work_dirs/v13-near${NW}-s0"
  for L in L8 L32 L256; do
    python scripts/eval_acc.py --manifest "manifests/acc_real_${L}.json" \
      --model "ours:$DIR/latest.pt" --tag "near${NW}-$L"
    python scripts/eval_acc.py --manifest "manifests/acc_real_${L}.json" \
      --model "ours:$DIR/latest.pt" --align scaleshift --per-frame \
      --tag "near${NW}-$L-pf"
  done
  python scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
    --model "ours:$DIR/latest.pt" --align scaleshift --regions \
    --tag "near${NW}-L8-regions"
done
echo "done -> scripts/acc_summary.py"
