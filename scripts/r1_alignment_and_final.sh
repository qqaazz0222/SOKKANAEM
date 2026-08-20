#!/usr/bin/env bash
# Two things the review asks for, chained so they never share the GPU:
#
#  A. every baseline under BOTH alignment rules (r1-9). The draft calls one
#     rule per model "the protocol", which is defensible but reads as a single
#     protocol when it is not; and it labels DA3 1-DOF when the script fits
#     scale AND shift. Measuring the whole group both ways settles it.
#  B. the final configuration by composition: the long-clip checkpoint plus
#     the spread term that fixes range compression, 8k steps at clip length 24
#     (r1-2, Section 6.5).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
DA3_PY=/home/hyunsu/miniforge3/envs/baselines/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
SRCS=("tum:$D/tum_static:walking_static"
      "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far")

while pgrep -f "r1_round.sh" > /dev/null; do
    echo "waiting for the measurement round..."; date -Is; sleep 300
done

echo "################ A. both alignment rules, 8-frame protocol ################"; date -Is
for MODE in relative metric; do
    for M in "DPT-Large:Intel/dpt-large" \
             "DA-v2-Base:depth-anything/Depth-Anything-V2-Base-hf" \
             "DA-v2-Small:depth-anything/Depth-Anything-V2-Small-hf" \
             "DA-v1-Small:LiheYoung/depth-anything-small-hf" \
             "ZoeDepth-NK:Intel/zoedepth-nyu-kitti"; do
        LABEL=${M%%:*}; CK=${M#*:}
        for SRC in "${SRCS[@]}"; do
            NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
            echo; echo "---- $LABEL / $NAME / align=$MODE ----"; date -Is
            EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $MODE" \
            MAX_CLIPS=100 \
                $PY scripts/eval_baseline_da2.py --ckpt "$CK" --label "$LABEL" \
                --mode "$MODE" 2>/dev/null | grep -E "pooled|holdout clips"
        done
    done
done
for ALIGN in scaleshift median; do
    for SRC in "${SRCS[@]}"; do
        NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
        echo; echo "---- DA3-Base / $NAME / align=$ALIGN ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME $ALIGN" \
        MAX_CLIPS=100 ALIGN=$ALIGN \
            $DA3_PY scripts/eval_baseline_da3.py 2>/dev/null \
            | grep -E "pooled|clipavg|holdout clips"
    done
done
echo; echo "---- ours, both rules, 8-frame ----"
for A in median scaleshift; do
    $PY scripts/eval.py --ckpt work_dirs/v9-60k/latest.pt "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 8 --max-clips 100 --align $A --scores-tag "align-$A"
done

echo; echo "################ B. long-clip + spread, 8k at clip length 24 ####"; date -Is
DIR=work_dirs/v11-longclip-spread
$PY scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/v10-longclip/latest.pt --resume-partial \
    --clip-len 24 --batch 2 --steps 8000 --seed 0 \
    --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
    --work-dir "$DIR"
for L in 8 32 256; do
    for K in 30 60; do
        STRIDE=$L; [ "$L" -ge 128 ] && STRIDE=64
        echo; echo "---- $DIR clip_len=$L keyframe=$K ----"; date -Is
        $PY scripts/eval.py --ckpt "$DIR/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
            --clip-len $L --clip-stride $STRIDE --max-clips 60 \
            --keyframe-every $K --control --scores-tag "L${L}K${K}"
    done
done
echo; echo "---- range ratio ----"; $PY scripts/range_probe.py --ckpt "$DIR/latest.pt"
echo; echo "---- frame-index, 256 frames, DISJOINT clips ----"
# disjoint: with stride 64 one outlier frame aliases into four indices, which
# is what produced the 64-periodic spikes in the round's curves
$PY scripts/frame_index_probe.py --ckpt "$DIR/latest.pt" --clip-len 256 \
    --max-clips 13 --every 16
echo "ALIGNMENT + FINAL CONFIG DONE"; date -Is
