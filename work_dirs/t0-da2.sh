#!/usr/bin/env bash
# DA2 leg of T0-1: the `baselines` env lost transformers, sokkanaem's has 5.14.1
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
export HF_HUB_OFFLINE=1
for SRC in "vkitti2:$D/vkitti2::Scene06" \
           "tartanair2:$D/tartanair_v2::OldTownFall" \
           "pointodyssey:$D/pointodyssey::/pointodyssey/val/,/pointodyssey/test/" \
           "tum:$D/tum_static::walking_static" \
           "bonn:$D/bonn/rgbd_bonn_dataset::rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far"; do
    SPEC=${SRC%%::*}; HOLD=${SRC##*::}; NAME=${SPEC%%:*}
    echo; echo "################ T0-1 baseline da2 — $NAME ################"; date -Is
    EVAL_SPECS="$SPEC" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" [$NAME]" MAX_CLIPS=100 \
        $PY scripts/eval_baseline_da2.py
done
echo "DA2 DONE"
