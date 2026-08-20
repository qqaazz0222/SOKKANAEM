#!/usr/bin/env bash
# Reviewer round 1 (paper/self-revision/r1.md). Serialized -- one GPU.
#
# Ordered by what the paper cannot ship without:
#  1. the comparison group at 8 frames, re-measured now that a clip cap
#     samples the whole holdout instead of its first sequence
#  2. Video Depth Anything, the missing video-specific baseline (r1-3)
#  3. our model and the group at 256 frames, the streaming protocol (r1-1)
#  4. frame-index curves on every metric, disjoint clips (r1-1)
#  5. both alignment rules for the whole group (r1-9)
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
SRCS=("tum:$D/tum_static:walking_static"
      "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far")
MODELS=("DPT-Large:Intel/dpt-large:relative"
        "DA-v2-Base:depth-anything/Depth-Anything-V2-Base-hf:relative"
        "DA-v2-Small:depth-anything/Depth-Anything-V2-Small-hf:relative"
        "DA-v1-Small:LiheYoung/depth-anything-small-hf:relative"
        "ZoeDepth-NK:Intel/zoedepth-nyu-kitti:metric")
V9=work_dirs/v9-60k/latest.pt
V10=work_dirs/v10-longclip/latest.pt

while pgrep -f "python.*scripts/train\.py" > /dev/null; do
    echo "waiting for a training run..."; date -Is; sleep 60
done

group() {   # group <clip_len> <max_clips> <label>
    local L=$1 MAXC=$2 TAG=$3
    for M in "${MODELS[@]}"; do
        local LABEL=${M%%:*}
        local REST=${M#*:}
        local CK=${REST%%:*}
        local MODE=${REST#*:}
        for SRC in "${SRCS[@]}"; do
            local NAME=${SRC%%:*}
            local R=${SRC#*:}
            local PATHP=${R%%:*}
            local HOLD=${R#*:}
            echo; echo "---- $LABEL / $NAME / clip_len=$L ----"; date -Is
            EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $TAG" \
            CLIP_LEN=$L MAX_CLIPS=$MAXC \
                $PY scripts/eval_baseline_da2.py --ckpt "$CK" --label "$LABEL" \
                --mode "$MODE" 2>/dev/null | grep -E "pooled|holdout clips"
        done
    done
    for SRC in "${SRCS[@]}"; do
        local NAME=${SRC%%:*}
        local R=${SRC#*:}
        local PATHP=${R%%:*}
        local HOLD=${R#*:}
        echo; echo "---- DA3-Base / $NAME / clip_len=$L ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $TAG" \
        CLIP_LEN=$L MAX_CLIPS=$MAXC ALIGN=scaleshift \
            $DA3_PY scripts/eval_baseline_da3.py 2>/dev/null \
            | grep -E "pooled|clipavg|holdout clips"
    done
}

vda() {   # vda <clip_len> <max_clips> <label>
    local L=$1 MAXC=$2 TAG=$3
    for SRC in "${SRCS[@]}" \
               "vkitti2:$D/vkitti2:Scene06" \
               "tartanair2:$D/tartanair_v2:OldTownFall" \
               "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
        local NAME=${SRC%%:*}
        local R=${SRC#*:}
        local PATHP=${R%%:*}
        local HOLD=${R#*:}
        echo; echo "---- VDA / $NAME / clip_len=$L ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $TAG" \
        CLIP_LEN=$L MAX_CLIPS=$MAXC \
            $VDA_PY scripts/eval_baseline_vda.py $VDA_DIR 2>/dev/null \
            | grep -E "pooled|clipavg|holdout clips"
    done
}

echo "################ 1. comparison group, 8 frames ################"; date -Is
group 8 100 L8

echo; echo "################ 2. Video Depth Anything ################"; date -Is
vda 8 100 L8

echo; echo "################ 3. streaming protocol, 256 frames ###########"; date -Is
for CK in $V9 $V10; do
    for L in 8 32 128 256; do
        for K in 30 60; do
            echo; echo "---- $CK clip_len=$L keyframe=$K ----"; date -Is
            $PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
                --clip-len $L --max-clips 100 --keyframe-every $K --control \
                --scores-tag "L${L}K${K}"
        done
    done
done
group 256 100 L256
vda 256 100 L256

echo; echo "################ 4. frame-index curves, disjoint clips #######"; date -Is
for CK in $V9 $V10; do
    for L in 32 256; do
        echo; echo "---- $CK, $L-frame ----"
        $PY scripts/frame_index_probe.py --ckpt $CK --clip-len $L \
            --max-clips 100 --every $((L / 16))
    done
done

echo; echo "################ 5. both alignment rules, 8 frames ###########"; date -Is
for M in "${MODELS[@]}"; do
    LABEL=${M%%:*}; REST=${M#*:}; CK=${REST%%:*}; MODE=${REST#*:}
    OTHER=metric; [ "$MODE" = metric ] && OTHER=relative
    for SRC in "${SRCS[@]}"; do
        NAME=${SRC%%:*}; R=${SRC#*:}; PATHP=${R%%:*}; HOLD=${R#*:}
        echo; echo "---- $LABEL / $NAME / align=$OTHER (non-native) ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $OTHER" \
        CLIP_LEN=8 MAX_CLIPS=100 \
            $PY scripts/eval_baseline_da2.py --ckpt "$CK" --label "$LABEL" \
            --mode "$OTHER" 2>/dev/null | grep -E "pooled|holdout clips"
    done
done
for SRC in "${SRCS[@]}"; do
    NAME=${SRC%%:*}; R=${SRC#*:}; PATHP=${R%%:*}; HOLD=${R#*:}
    echo; echo "---- DA3-Base / $NAME / align=median (non-native) ----"; date -Is
    EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME median" \
    CLIP_LEN=8 MAX_CLIPS=100 ALIGN=median \
        $DA3_PY scripts/eval_baseline_da3.py 2>/dev/null \
        | grep -E "pooled|clipavg|holdout clips"
done
for A in median scaleshift; do
    echo; echo "---- ours / align=$A ----"; date -Is
    $PY scripts/eval.py --ckpt $V9 "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 8 --max-clips 100 --align $A --scores-tag "align-$A"
done

echo; echo "R1 ROUND DONE"; date -Is
