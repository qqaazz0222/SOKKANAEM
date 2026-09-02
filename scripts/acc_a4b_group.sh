#!/usr/bin/env bash
# The rest of the comparison group, on the sealed manifests.
#
# A4 measured only DPT-Large and DA V2 Small, because those are what G1 is
# written against. The repo has been quoting five more models from the old
# even_subset protocol, and those numbers cannot enter the same table as these
# ones -- different clip set, different alignment, different failure handling.
# This puts the whole group on one protocol.
#
# Video-Depth-Anything is missing: its checkout lived in a scratchpad that is
# gone, and it is the one model here that is not a Hugging Face AutoModel.
# DA3 runs from the `baselines` env (scripts/acc_a4b_da3.sh).
set -eu
cd "$(dirname "$0")/.."

MODELS="${MODELS:-LiheYoung/depth-anything-small-hf
                  depth-anything/Depth-Anything-V2-Base-hf
                  Intel/zoedepth-nyu-kitti}"

for B in $MODELS; do
  N="$(basename "$B")"
  for L in ${MANIFESTS:-L8 L32 L256}; do
    M="manifests/acc_real_${L}.json"
    for SZ in official 256; do
      echo "=== $M : $N @ $SZ"
      ARG=""; [ "$SZ" = 256 ] && ARG="--infer-size 256"
      python scripts/eval_acc.py --manifest "$M" --model "hf:$B" \
        $ARG --tag "$N-$L-$SZ"
    done
    echo "=== $M : $N per-frame"
    python scripts/eval_acc.py --manifest "$M" --model "hf:$B" \
      --align scaleshift --per-frame --tag "$N-$L-official-pf"
  done
done
echo "done -> scripts/acc_summary.py"
