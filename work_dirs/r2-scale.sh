#!/usr/bin/env bash
# r2 Major 1: the per-frame scale factor s_t for the comparison group.
#
# The statistics (CV, log-std, frame-to-frame |log s_t - log s_{t-1}|) now come
# out of the shared scorer (sokkanaem/metrics.py:scale_stats), so this only has
# to re-run the models -- no new metric code per baseline. Real indoor holdout
# only: that is the domain Table 3a/7f compare on.
#
# Needs the GPU to itself. Run it after the training job in work_dirs/ finishes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
CKPT=work_dirs/v11-longclip-spread-s0/latest.pt
export MAX_CLIPS=100

for L in 8 32 256; do
    echo; echo "======== ours L$L ========"; date -Is
    $PY scripts/eval.py --ckpt "$CKPT" \
        --data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset" \
        --holdout walking_static --holdout rgbd_bonn_crowd2 \
        --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far \
        --clip-len "$L" --size 256 --keyframe-every 30 --scores-tag "L${L}K30"
done

# baselines: same holdout, same alignment rules as work_dirs/baseline-suite.sh
base() {  # base <label> <hf-id> <mode>
    for SRC in "tum:$D/tum_static:walking_static" \
               "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far"; do
        NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
        for L in 8 256; do
            echo; echo "---- $1 / $NAME / L$L ----"; date -Is
            CLIP_LEN=$L EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" \
                EVAL_TAG=" $NAME l$L" \
                $PY scripts/eval_baseline_da2.py --ckpt "$2" --label "$1" --mode "$3" \
                2>/dev/null | grep -E "pooled|holdout clips|per-clip"
        done
    done
}

base "DPT-Large"   Intel/dpt-large                          relative
base "DA-v1-Small" LiheYoung/depth-anything-small-hf         relative
base "DA-v2-Base"  depth-anything/Depth-Anything-V2-Base-hf  relative
base "ZoeDepth-NK" Intel/zoedepth-nyu-kitti                  metric

# DA3 and VDA live in their own envs -- see the header of each script for the
# interpreter, and pass the same CLIP_LEN / EVAL_* variables.

echo; echo "R2 SCALE SUITE DONE"; date -Is
echo "then: $PY scripts/bootstrap_ci.py --ours work_dirs/v11-longclip-spread-s0/scores_L256K30.json \\"
echo "        --suffix l256 --metric scale_logstd --metric scale_step --metric scale_drift"
