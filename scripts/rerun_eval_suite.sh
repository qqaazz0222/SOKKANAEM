#!/usr/bin/env bash
# Re-run every reported evaluation under one protocol (REPORT.md §4.10/§4.13):
# 1000 clips (not 100), pixel-pooled metrics, OPW+TCE, constant-depth control,
# per-clip values dumped to JSON. Serialized on purpose — the GPU is shared
# with a training run, and parallel evals just make every number arrive later.
#
# Usage: scripts/rerun_eval_suite.sh [work_dir_of_ckpt ...]
set -uo pipefail
cd "$(dirname "$0")/.."

MIX=(--data vkitti2:/home/hyunsu/dataset_ssd/vkitti2
     --data tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2
     --data pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey)
HOLD=(--holdout Scene06 --holdout OldTownFall
      --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
if [ $# -gt 0 ]; then
    CKPTS=("$@")
else
    CKPTS=(work_dirs/main-v3-nodist-20260725 work_dirs/main-v4-distill-20260726)
fi
RUN="conda run -n sokkanaem --no-capture-output python"

for d in "${CKPTS[@]}"; do
    echo "################ $d — holdout tau sweep (delta) ################"
    $RUN scripts/eval.py --ckpt "$d/latest.pt" "${MIX[@]}" "${HOLD[@]}" \
        --size 256 --clip-len 8 --max-clips 1000 --sweep-tau --control --scores-tag holdout
    echo "################ $d — gating position: token drop ################"
    $RUN scripts/eval.py --ckpt "$d/latest.pt" "${MIX[@]}" "${HOLD[@]}" \
        --size 256 --clip-len 8 --max-clips 1000 --sweep-tau --gate-mode drop --scores-tag holdout-drop
    echo "################ $d — long clips (32 frames) ################"
    # clip evaluation resets the keyframe every clip, which badly understates
    # the skip ratio (IDEA.md §4.5: streaming 20-28% vs clip 68%). 32-frame
    # clips are the deployment-side number.
    $RUN scripts/eval.py --ckpt "$d/latest.pt" "${MIX[@]}" "${HOLD[@]}" \
        --size 256 --clip-len 32 --max-clips 250 --sweep-tau --scores-tag longclip
    echo "################ $d — real fixed camera (TUM fr3 static) ########"
    $RUN scripts/eval.py --ckpt "$d/latest.pt" \
        --data tum:/home/hyunsu/dataset_ssd/tum_static \
        --size 256 --clip-len 8 --max-clips 1000 --sweep-tau --control \
        --scores-tag tum-static
done

echo "################ external baselines, same 1000 clips ################"
$RUN scripts/eval_baseline_da2.py
conda run -n baselines --no-capture-output python scripts/eval_baseline_da3.py
conda run -n vda --no-capture-output python scripts/eval_baseline_vda.py \
    /tmp/claude-1000/-workspace-SOKKANAEM/58ef1dd9-33b5-4c0a-a578-5561d3855ced/scratchpad/Video-Depth-Anything
echo "################ done ################"
