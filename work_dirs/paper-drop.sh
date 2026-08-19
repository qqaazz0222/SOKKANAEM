#!/usr/bin/env bash
# Table 4 on the confirmed checkpoint: Delta-gating vs token drop at matched
# masks. Both arms run with the temporal cache OFF -- the drop ablation exists
# to isolate whether READING the preserved state matters, so the arm it is
# compared against must still perform that dense readout.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
D=/home/hyunsu/dataset_ssd
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
for MODE in delta drop; do
    echo; echo "################ gate_mode=$MODE ################"; date -Is
    $PY scripts/eval.py --ckpt $CK "${SYNTH[@]}" "${SHOLD[@]}" \
        --max-clips 100 --no-temporal-cache --gate-mode $MODE \
        --scores-tag "paper-$MODE"
done
echo "DROP ABLATION DONE"; date -Is
