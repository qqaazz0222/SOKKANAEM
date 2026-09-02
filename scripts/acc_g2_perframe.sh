#!/usr/bin/env bash
# G2's other half: the same clips scored with the gauge fitted per FRAME.
#
# Under the clip-wide fit a stateless DPT-Large loses 116% from 8-frame to
# 256-frame clips. It has no state, so that growth cannot be drift -- it is one
# scale+shift failing to cover a longer stream. PLAN_ACC §1.3 asks for both
# numbers for exactly this reason, and only the per-frame one is a statement
# about the model.
set -eu
cd "$(dirname "$0")/.."

MANIFESTS="${MANIFESTS:-L8 L32 L256}"
OURS="${OURS:-work_dirs/v11-longclip-spread-s0/latest.pt}"

for L in $MANIFESTS; do
  M="manifests/acc_real_${L}.json"
  echo "=== $M : ours per-frame"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --align scaleshift --per-frame --tag "ours-$L-pf"
  echo "=== $M : dpt-large per-frame"
  python scripts/eval_acc.py --manifest "$M" --model hf:Intel/dpt-large \
    --align scaleshift --per-frame --tag "dpt-large-$L-official-pf"
done
echo "done -> work_dirs/acc/table.txt"
