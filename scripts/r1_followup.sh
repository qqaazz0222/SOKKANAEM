#!/usr/bin/env bash
# Queued behind scripts/r1_round.sh.
#
#  1. re-score the three spread-term arms (Section 6.5, Table 11) under the
#     fixed clip sampling -- all three shared the biased cap, so their ranking
#     stands, but the absolute numbers move
#  2. the final configuration by composition: long-clip checkpoint + spread
#     term, 8k steps at clip length 24 (r1-2). Retraining from scratch is
#     18 h 45 min; this stage is 8k steps.
#  3. two more seeds of that stage, so the final numbers carry a spread
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

while pgrep -f "bash scripts/r1_round\.sh" > /dev/null; do
    echo "waiting for the measurement round..."; date -Is; sleep 300
done

echo "################ 1. spread arms, fixed sampling ################"; date -Is
for W in 0.0 0.5 2.0; do
    echo; echo "---- spread_weight=$W ----"; date -Is
    $PY scripts/eval.py --ckpt "work_dirs/t5-spread$W/latest.pt" "${REAL[@]}" \
        "${RHOLD[@]}" --clip-len 8 --max-clips 100 --scores-tag even-real
    $PY scripts/range_probe.py --ckpt "work_dirs/t5-spread$W/latest.pt"
done

echo; echo "################ 2. synthetic domain, fixed sampling ##########"; date -Is
# the synthetic caps were the most biased of all: PointOdyssey offers 1,984
# clips at 8 frames and the cap took the first 100, i.e. its first held-out
# sequence. Ours, the token-drop control, and the baseline group all move.
SYN=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
     --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
for CK in work_dirs/v9-60k/latest.pt work_dirs/v10-longclip/latest.pt; do
    echo; echo "---- $CK / synthetic sweep ----"; date -Is
    $PY scripts/eval.py --ckpt $CK "${SYN[@]}" "${SHOLD[@]}" \
        --clip-len 8 --max-clips 100 --sweep-tau --control --scores-tag even-synth
    echo; echo "---- $CK / synthetic, token drop ----"; date -Is
    $PY scripts/eval.py --ckpt $CK "${SYN[@]}" "${SHOLD[@]}" \
        --clip-len 8 --max-clips 100 --gate-mode drop \
        --no-temporal-cache --scores-tag even-synth-drop
    echo; echo "---- $CK / synthetic, delta gating, cache off ----"; date -Is
    $PY scripts/eval.py --ckpt $CK "${SYN[@]}" "${SHOLD[@]}" \
        --clip-len 8 --max-clips 100 --no-temporal-cache \
        --scores-tag even-synth-delta
    echo; echo "---- $CK / real sweep ----"; date -Is
    $PY scripts/eval.py --ckpt $CK "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 8 --max-clips 100 --sweep-tau --control --scores-tag even-real-sweep
done
for M in "DPT-Large:Intel/dpt-large:relative" \
         "DA-v2-Base:depth-anything/Depth-Anything-V2-Base-hf:relative" \
         "DA-v2-Small:depth-anything/Depth-Anything-V2-Small-hf:relative" \
         "DA-v1-Small:LiheYoung/depth-anything-small-hf:relative" \
         "ZoeDepth-NK:Intel/zoedepth-nyu-kitti:metric"; do
    LABEL=${M%%:*}; REST=${M#*:}; CK=${REST%%:*}; MODE=${REST#*:}
    for SRC in "vkitti2:$D/vkitti2:Scene06" \
               "tartanair2:$D/tartanair_v2:OldTownFall" \
               "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
        NAME=${SRC%%:*}; R=${SRC#*:}; PATHP=${R%%:*}; HOLD=${R#*:}
        echo; echo "---- $LABEL / $NAME ----"; date -Is
        EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME even" \
        CLIP_LEN=8 MAX_CLIPS=100 \
            $PY scripts/eval_baseline_da2.py --ckpt "$CK" --label "$LABEL" \
            --mode "$MODE" 2>/dev/null | grep -E "pooled|holdout clips"
    done
done
for SRC in "vkitti2:$D/vkitti2:Scene06" \
           "tartanair2:$D/tartanair_v2:OldTownFall" \
           "pointodyssey:$D/pointodyssey:/pointodyssey/val/,/pointodyssey/test/"; do
    NAME=${SRC%%:*}; R=${SRC#*:}; PATHP=${R%%:*}; HOLD=${R#*:}
    echo; echo "---- DA3-Base / $NAME ----"; date -Is
    EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" $NAME even" \
    CLIP_LEN=8 MAX_CLIPS=100 ALIGN=scaleshift \
        /home/hyunsu/miniforge3/envs/baselines/bin/python \
        scripts/eval_baseline_da3.py 2>/dev/null \
        | grep -E "pooled|clipavg|holdout clips"
done

echo; echo "################ 3. long-clip + spread, three seeds ###########"; date -Is
for SEED in 0 1 2; do
    DIR=work_dirs/v11-longclip-spread-s$SEED
    echo; echo "---- seed $SEED ----"; date -Is
    $PY scripts/train.py --config configs/main_v8.toml \
        --resume work_dirs/v10-longclip/latest.pt --resume-partial \
        --clip-len 24 --batch 2 --steps 8000 --seed $SEED \
        --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
        --work-dir "$DIR"
    for L in 8 32 256; do
        for K in 30 60; do
            echo; echo "---- seed $SEED clip_len=$L keyframe=$K ----"; date -Is
            $PY scripts/eval.py --ckpt "$DIR/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
                --clip-len $L --max-clips 100 --keyframe-every $K --control \
                --scores-tag "L${L}K${K}"
        done
    done
    $PY scripts/range_probe.py --ckpt "$DIR/latest.pt"
    $PY scripts/frame_index_probe.py --ckpt "$DIR/latest.pt" --clip-len 256 \
        --max-clips 100 --every 16
done
echo "FOLLOWUP DONE"; date -Is
