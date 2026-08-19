#!/usr/bin/env bash
# T5-24: train on long clips so the model actually sees sustained gating.
#
# Training used clip_len=4 while deployment runs hundreds of frames, so the
# model had never had to survive more than three consecutive gated frames.
# REPORT 4.30 measured the consequence: on Bonn, AbsRel degrades 61% between
# keyframes. clip_len 24 with keyframe_every 30 means no mid-clip refresh --
# the model must hold up for 23 gated frames in a row.
#
# Fine-tune from the confirmed checkpoint rather than retraining: this is an
# adaptation to a different temporal regime, not a fresh optimisation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd

$PY scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/v9-60k/latest.pt --resume-partial \
    --clip-len 24 --batch 2 --steps 25000 --seed 0 \
    --warp-weight 2.0 --edge-weight 2.0 \
    --work-dir work_dirs/v10-longclip

REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)

echo; echo "######## 8-frame protocol (comparable to every reported number) ########"
$PY scripts/eval.py --ckpt work_dirs/v10-longclip/latest.pt "${REAL[@]}" "${RHOLD[@]}" \
    --max-clips 100 --scores-tag real
$PY scripts/eval.py --ckpt work_dirs/v10-longclip/latest.pt "${SYNTH[@]}" "${SHOLD[@]}" \
    --max-clips 100 --scores-tag synth

echo; echo "######## 32-frame protocol (the setting this run targets) ########"
$PY scripts/eval.py --ckpt work_dirs/v10-longclip/latest.pt "${REAL[@]}" "${RHOLD[@]}" \
    --clip-len 32 --max-clips 30 --scores-tag real32

echo "T5-24 DONE"; date -Is
