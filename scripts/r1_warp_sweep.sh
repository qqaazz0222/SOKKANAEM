#!/usr/bin/env bash
# Does the flow-warped residual term buy OPW/TCE, and at what cost in raw frame
# difference? Only worth asking now: the final stage's three-seed spread is
# 0.0012 AbsRel and 0.0002 OPW, so a 3% change is resolvable where under the
# old 8k-derived noise floor it was not.
#
# One variable. Weight 2.0 is the reported configuration (v11-*-s0), so only
# the two higher weights need training.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

for W in 4.0 8.0; do
    DIR=work_dirs/v11-warp$W
    echo; echo "################ warp_weight=$W ################"; date -Is
    $PY scripts/train.py --config configs/main_v8.toml \
        --resume work_dirs/v10-longclip/latest.pt --resume-partial \
        --clip-len 24 --batch 2 --steps 8000 --seed 0 \
        --warp-weight $W --edge-weight 2.0 --spread-weight 0.5 \
        --work-dir "$DIR"
    for L in 8 32 256; do
        for K in 30 60; do
            echo; echo "---- warp=$W clip_len=$L K=$K ----"; date -Is
            $PY scripts/eval.py --ckpt "$DIR/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
                --clip-len $L --max-clips 100 --keyframe-every $K \
                --scores-tag "warp${W}-L${L}K${K}"
        done
    done
    $PY scripts/range_probe.py --ckpt "$DIR/latest.pt" --max-clips 100
done
echo "WARP SWEEP DONE"; date -Is
