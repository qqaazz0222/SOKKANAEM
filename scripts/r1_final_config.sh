#!/usr/bin/env bash
# The final configuration the review asks for (r1-2), built by composition
# rather than by retraining: the reported checkpoint already has a long-clip
# stage (v10, 25k at clip length 24, 18 h 45 min), and the spread term that
# fixes range compression (Section 6.5, Table 12) is an 8k stage. Stacking
# them costs 8k steps, not another 25k.
#
# The spread stage runs at clip length 24, not the config default of 4: a
# clip-4 stage on top of a long-clip checkpoint would spend 8k steps undoing
# the property the long-clip stage bought.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

while pgrep -f "r1_round.sh" > /dev/null; do
    echo "waiting for the measurement round..."; date -Is; sleep 300
done

DIR=work_dirs/v11-longclip-spread
echo "################ long-clip + spread, 8k at clip length 24 ################"; date -Is
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
echo; echo "---- range ratio ----"
$PY scripts/range_probe.py --ckpt "$DIR/latest.pt"
echo; echo "---- frame-index, 256 frames ----"
$PY scripts/frame_index_probe.py --ckpt "$DIR/latest.pt" --clip-len 256 \
    --clip-stride 64 --max-clips 12 --every 16
echo "FINAL CONFIG DONE"; date -Is
