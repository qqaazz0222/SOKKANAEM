#!/usr/bin/env bash
# The commonly cited comparison group, all under OUR protocol.
#
# Published numbers for these models come from different splits, resolutions
# and alignment rules, so quoting them side by side would not be a comparison.
# Every row here runs the same holdout clips, the same 256px input, the same
# per-clip alignment and the same metric implementation as scripts/eval.py.
#
# Relative-depth models get least-squares scale+shift in disparity space (the
# MiDaS protocol they are designed for); the metric model gets the per-clip
# median scaling our own eval uses.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
export MAX_CLIPS=100

run() {   # run <label> <ckpt> <mode>
    for SRC in "tum:$D/tum_static:walking_static" \
               "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far" \
               "vkitti2:$D/vkitti2:Scene06" \
               "tartanair2:$D/tartanair_v2:OldTownFall" \
               "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
        NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
        echo; echo "---- $1 / $NAME ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME" \
            $PY scripts/eval_baseline_da2.py --ckpt "$2" --label "$1" --mode "$3" \
            2>/dev/null | grep -E "pooled|holdout clips"
    done
}

run "DPT-Large"   Intel/dpt-large                            relative
run "DA-v2-Base"  depth-anything/Depth-Anything-V2-Base-hf   relative
run "DA-v1-Small" LiheYoung/depth-anything-small-hf          relative
run "ZoeDepth-NK" Intel/zoedepth-nyu-kitti                   metric

echo; echo "BASELINE SUITE DONE"; date -Is
