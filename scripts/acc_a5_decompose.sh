#!/usr/bin/env bash
# PLAN_ACC A5: split the reported error between the spatial predictor and the
# temporal state, on the sealed manifests.
#
# The reported number is one figure covering two mechanisms, and every fix so
# far has been aimed without knowing which one owns it. Four modes of the same
# checkpoint answer that:
#
#   sparse    shipped configuration                      (the reported number)
#   dense     tau=0, every patch active                  (what gating costs)
#   reset     fresh state each frame                     (spatial predictor alone)
#   notemp    TemporalBlocks made identity               (the blocks' own worth)
#
# `sparse` rows are already produced by acc_phase_a.sh, so this runs the other
# three. PLAN_ACC §2 already records that tau=0 barely moves the 8-frame number
# (0.123 against 0.1263) -- the open question is the 256-frame protocol, where
# the error grows 51% and the state is the only thing that changed.
set -eu
cd "$(dirname "$0")/.."

MANIFESTS="${MANIFESTS:-L8 L32 L256}"
OURS="${OURS:-work_dirs/v11-longclip-spread-s0/latest.pt}"

for L in $MANIFESTS; do
  M="manifests/acc_real_${L}.json"
  echo "=== $M : dense (tau=0)"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --tau 0 --tag "ours-$L-dense"
  echo "=== $M : reset every frame"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --reset-every 1 --tag "ours-$L-reset1"
  echo "=== $M : temporal blocks bypassed"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --bypass-temporal --tag "ours-$L-notemp"
done
echo "done -> work_dirs/acc/table.txt"
