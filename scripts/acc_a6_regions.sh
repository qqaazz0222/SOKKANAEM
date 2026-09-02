#!/usr/bin/env bash
# PLAN_ACC A6: where the error lives, per pixel rather than per clip.
#
# Only the primary gauge (scaleshift, G1's) and only the two rows the gate is
# written against -- ours and the reference. The region masks come from GT
# depth and RGB alone, so the two rows partition the same pixels and the
# difference between them is the model.
#
# One RAFT pass per clip dominates the cost (~15 min per row at L8), which is
# why this is a separate script rather than a flag on the A4 matrix.
set -eu
cd "$(dirname "$0")/.."

MANIFESTS="${MANIFESTS:-L8 L256}"
OURS="${OURS:-work_dirs/v11-longclip-spread-s0/latest.pt}"

for L in $MANIFESTS; do
  M="manifests/acc_real_${L}.json"
  echo "=== $M : ours regions"
  python scripts/eval_acc.py --manifest "$M" --model "ours:$OURS" \
    --align scaleshift --regions --tag "ours-$L-regions"
  echo "=== $M : dpt-large regions"
  python scripts/eval_acc.py --manifest "$M" --model hf:Intel/dpt-large \
    --align scaleshift --regions --tag "dpt-large-$L-regions"
done
echo "done -> work_dirs/acc/table.txt"
