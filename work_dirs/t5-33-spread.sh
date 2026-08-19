#!/usr/bin/env bash
# T5-33: does penalising range compression actually widen the prediction?
#   setsid nohup bash work_dirs/t5-33-spread.sh > work_dirs/t5-33-spread.log 2>&1 < /dev/null &
#
# Queued behind T5-24 so the two runs do not share the GPU. Three weights on a
# short probe, all from the same init and the same seed, because the question
# here is direction and dose, not final accuracy.
#
# The failure mode to watch for is the term being bought with noise: spread
# rises, AbsRel does not improve or gets worse. The probe therefore reports the
# range ratio alongside accuracy rather than the loss value.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)

while pgrep -f 't5-longclip.sh' > /dev/null; do sleep 300; done
echo "T5-24 finished, starting spread probe"; date -Is

for W in 0.0 0.5 2.0; do
    DIR=work_dirs/t5-spread$W
    echo; echo "################ spread_weight=$W ################"; date -Is
    $PY scripts/train.py --config configs/main_v8.toml \
        --resume work_dirs/v9-60k/latest.pt --resume-partial \
        --steps 8000 --seed 0 --warp-weight 2.0 --edge-weight 2.0 \
        --spread-weight $W --work-dir "$DIR"
    $PY scripts/eval.py --ckpt "$DIR/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
        --max-clips 100 --scores-tag real
    echo "-- range ratio --"
    CKPT="$DIR/latest.pt" $PY scripts/range_probe.py --ckpt "$DIR/latest.pt"
done
echo "T5-33 DONE"; date -Is
