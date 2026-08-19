#!/usr/bin/env bash
# Regenerate every paper table on ONE checkpoint.
#
# The draft currently mixes three: Table 1 is v7, Tables 3 and 4 are v3, and
# Tables 2/3b/5/6 are the confirmed v9. A reviewer's first question would be
# which model the paper is about. These two runs put the activity sweep and the
# gating-location ablation on the confirmed checkpoint so every number in the
# paper comes from the same weights.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
D=/home/hyunsu/dataset_ssd
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

echo "################ Table 1: activity sweep, synthetic ################"; date -Is
$PY scripts/eval.py --ckpt $CK "${SYNTH[@]}" "${SHOLD[@]}" \
    --max-clips 100 --sweep-tau --control

echo; echo "################ Table 1b: activity sweep, real ################"; date -Is
$PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
    --max-clips 100 --sweep-tau --control

echo; echo "################ Table 4: token-drop control ################"; date -Is
$PY scripts/eval.py --ckpt $CK "${SYNTH[@]}" "${SHOLD[@]}" \
    --max-clips 100 --gate-mode drop --scores-tag paper-drop

echo "PAPER TABLES DONE"; date -Is
