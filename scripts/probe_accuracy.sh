#!/usr/bin/env bash
# Accuracy recipe probes: train 3 short arms, score each on the same holdout.
# Sequential — one GPU, and two concurrent trainings already crashed once
# (SIGILL/SIGSEGV, 2026-07-26).
#
#   p0-noaug  current recipe                       (reference)
#   p1-aug    + clip-consistent crop/flip/colour   (generalization)
#   p2-dpt    + DPT decoder + disparity head + multi-scale gradient loss
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="conda run -n sokkanaem --no-capture-output python"

train_arm() {  # name, config, extra flags...
    local name=$1 cfg=$2; shift 2
    echo "################ TRAIN $name ################"
    $RUN scripts/train.py --config "$cfg" --work-dir "work_dirs/$name" "$@"
    echo "################ EVAL $name (synthetic holdout) ################"
    $RUN scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        --data vkitti2:/home/hyunsu/dataset_ssd/vkitti2 \
        --data tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2 \
        --data pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey \
        --holdout Scene06 --holdout OldTownFall \
        --holdout /pointodyssey/val/ --holdout /pointodyssey/test/ \
        --size 128 --clip-len 8 --max-clips 400 --sweep-tau --control \
        --scores-tag holdout
    echo "################ EVAL $name (real unseen) ################"
    $RUN scripts/eval.py --ckpt "work_dirs/$name/latest.pt" \
        --data tum:/home/hyunsu/dataset_ssd/tum_static \
        --data bonn:/home/hyunsu/dataset_ssd/bonn/rgbd_bonn_dataset \
        --holdout walking_static --holdout rgbd_bonn_crowd2 \
        --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far \
        --size 128 --clip-len 8 --max-clips 400 --sweep-tau --control \
        --scores-tag real-unseen
}

train_arm probe-p0-noaug configs/probe_base.toml --no-augment
train_arm probe-p1-aug   configs/probe_base.toml
train_arm probe-p2-dpt   configs/probe_dpt.toml --msgrad-weight 0.5
echo "################ probes done ################"
