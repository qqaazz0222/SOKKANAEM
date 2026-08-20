#!/usr/bin/env bash
# Reviewer round 1 (paper/self-revision/r1.md): the measurements the review
# asks for, in value order. Serialized -- one GPU, and parallel runs only make
# every number arrive later.
#
#   1. frame-index curves on ALL metrics, both checkpoints (r1-1)
#   2. our model under a 256-frame streaming protocol (r1-1, r1-2)
#   3. Video Depth Anything, the missing video-specific baseline (r1-3)
#   4. the rest of the comparison group at the same clip length (r1-1, r1-9)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
VDA_PY=/home/hyunsu/miniforge3/envs/vda/bin/python
DA3_PY=/home/hyunsu/miniforge3/envs/baselines/bin/python
VDA_DIR=/home/hyunsu/checkouts/Video-Depth-Anything
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
V9=work_dirs/v9-60k/latest.pt
V10=work_dirs/v10-longclip/latest.pt

# never share the GPU with a training run
while pgrep -f "scripts/train.py" > /dev/null; do
    echo "waiting for the training run to finish..."; date -Is; sleep 60
done

echo "################ 1. frame-index curves, all metrics ################"; date -Is
# 32 frames first: regenerates Table 7, whose raw output was never saved to a
# log (only REPORT.md carried it)
for CK in $V9 $V10; do
    echo; echo "---- $CK, 32-frame ----"
    $PY scripts/frame_index_probe.py --ckpt $CK --clip-len 32 --max-clips 30 --every 4
    echo; echo "---- $CK, 256-frame, stride 64 ----"
    $PY scripts/frame_index_probe.py --ckpt $CK --clip-len 256 --clip-stride 64 \
        --max-clips 12 --every 16
done

echo; echo "################ 2. our model, clip-length ladder ################"; date -Is
for CK in $V9 $V10; do
    for L in 8 32 128 256; do
        for K in 30 60; do
            echo; echo "---- $CK clip_len=$L keyframe=$K ----"; date -Is
            STRIDE=$L; [ "$L" -ge 128 ] && STRIDE=64
            $PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
                --clip-len $L --clip-stride $STRIDE --max-clips 60 \
                --keyframe-every $K --control \
                --scores-tag "L${L}K${K}"
        done
    done
done

echo; echo "################ 3. Video Depth Anything ################"; date -Is
for SRC in "tum:$D/tum_static:walking_static" \
           "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far" \
           "vkitti2:$D/vkitti2:Scene06" \
           "tartanair2:$D/tartanair_v2:OldTownFall" \
           "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
    NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
    for L in 8 256; do
        MAXC=100; STRIDE=$L
        [ "$L" = 256 ] && { MAXC=60; STRIDE=64; }
        echo; echo "---- VDA / $NAME / clip_len=$L ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME L$L" \
        CLIP_LEN=$L CLIP_STRIDE=$STRIDE MAX_CLIPS=$MAXC \
            $VDA_PY scripts/eval_baseline_vda.py $VDA_DIR 2>/dev/null \
            | grep -E "pooled|clipavg|holdout clips"
    done
done

echo; echo "################ 4. comparison group at 256 frames ################"; date -Is
run_da2() {   # run_da2 <label> <ckpt> <mode>
    for SRC in "tum:$D/tum_static:walking_static" \
               "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far"; do
        NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
        echo; echo "---- $1 / $NAME ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME" \
        CLIP_LEN=256 CLIP_STRIDE=64 MAX_CLIPS=60 \
            $PY scripts/eval_baseline_da2.py --ckpt "$2" --label "$1" --mode "$3" \
            2>/dev/null | grep -E "pooled|holdout clips"
    done
}
run_da2 "DPT-Large"    Intel/dpt-large                            relative
run_da2 "DA-v2-Base"   depth-anything/Depth-Anything-V2-Base-hf   relative
run_da2 "DA-v2-Small"  depth-anything/Depth-Anything-V2-Small-hf  relative
run_da2 "DA-v1-Small"  LiheYoung/depth-anything-small-hf          relative
run_da2 "ZoeDepth-NK"  Intel/zoedepth-nyu-kitti                   metric

for SRC in "tum:$D/tum_static:walking_static" \
           "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far"; do
    NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
    echo; echo "---- DA3-Base / $NAME ----"; date -Is
    EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME" \
    CLIP_LEN=256 CLIP_STRIDE=64 MAX_CLIPS=60 \
        $DA3_PY scripts/eval_baseline_da3.py 2>/dev/null \
        | grep -E "pooled|clipavg|holdout clips"
done

echo; echo "R1 ROUND DONE"; date -Is
