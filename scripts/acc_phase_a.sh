#!/usr/bin/env bash
# PLAN_ACC A4: the comparison group re-measured on the sealed manifests.
#
# Every row shares manifest, alignment, scorer and statistics -- only the model
# differs. Baselines run twice: at their official input resolution (the quality
# ceiling) and at the manifest's 256 (the like-for-like row G1 is written
# against), because a table that quotes only one of those is choosing which
# claim to support.
set -eu
cd "$(dirname "$0")/.."

MANIFESTS="${MANIFESTS:-L8 L32 L256}"
OURS="${OURS:-work_dirs/v11-longclip-spread-s0/latest.pt}"
BASELINES="${BASELINES:-Intel/dpt-large depth-anything/Depth-Anything-V2-Small-hf}"

for L in $MANIFESTS; do
  M="manifests/acc_real_${L}.json"
  echo "=== $M : ours"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --tag "ours-$L"
  for B in $BASELINES; do
    N="$(basename "$B")"
    for SZ in official 256; do
      echo "=== $M : $N @ $SZ"
      ARG=""; [ "$SZ" = 256 ] && ARG="--infer-size 256"
      python scripts/eval_acc.py --manifest "$M" --model "hf:$B" \
        $ARG --tag "$N-$L-$SZ"
    done
  done
done
echo "done -> work_dirs/acc/table.txt"
