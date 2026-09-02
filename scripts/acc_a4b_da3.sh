#!/usr/bin/env bash
# DA3-Base on the sealed manifests. Separate script because Depth Anything 3
# is not a Hugging Face AutoModel and lives in the `baselines` conda env.
set -eu
cd "$(dirname "$0")/.."

CKPT="${CKPT:-depth-anything/DA3-BASE}"
RUN="conda run -n baselines --no-capture-output python"

for L in ${MANIFESTS:-L8 L32 L256}; do
  M="manifests/acc_real_${L}.json"
  echo "=== $M : DA3-BASE"
  $RUN scripts/eval_acc.py --manifest "$M" --model "da3:$CKPT" \
    --tag "DA3-BASE-$L-official"
  echo "=== $M : DA3-BASE per-frame"
  $RUN scripts/eval_acc.py --manifest "$M" --model "da3:$CKPT" \
    --align scaleshift --per-frame --tag "DA3-BASE-$L-official-pf"
done
echo "done"
