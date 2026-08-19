#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
export MAX_CLIPS=100
for SRC in "tum:$D/tum_static:walking_static" \
           "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far" \
           "vkitti2:$D/vkitti2:Scene06" \
           "tartanair2:$D/tartanair_v2:OldTownFall" \
           "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
    NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
    echo; echo "---- ZoeDepth-NK / $NAME ----"
    EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME" \
        $PY scripts/eval_baseline_da2.py --ckpt Intel/zoedepth-nyu-kitti \
        --label ZoeDepth-NK --mode metric 2>/dev/null | grep -E "pooled"
done
echo "ZOE DONE"
