#!/usr/bin/env bash
# PLAN.md Tier 0, run as one detached queue (setsid nohup) so it survives the
# session that started it. Cheap items first, the 10 h seed sweep last.
#   setsid nohup bash work_dirs/tier0.sh > work_dirs/tier0-console.log 2>&1 < /dev/null &
set -uo pipefail          # NOT -e: one broken stage must not kill the queue
cd "$(dirname "$0")/.."

PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
CKPT=work_dirs/v8-teacherfree-60k/latest.pt
SYNTH=(--data "vkitti2:$D/vkitti2" --data "tartanair2:$D/tartanair_v2"
       --data "pointodyssey:$D/pointodyssey")
SHOLD=(--holdout Scene06 --holdout OldTownFall
       --holdout /pointodyssey/val/ --holdout /pointodyssey/test/)
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
export HF_HUB_OFFLINE=1   # every baseline weight is already in ~/.cache/huggingface

step() { echo; echo "################ $* ################"; date -Is; }

# ---- T0-3 verification: the dense-fallback policy on the real checkpoint ----
step "T0-3 dense_above policy — synthetic (TartanAir sits at 70% active)"
$PY scripts/eval.py --ckpt $CKPT "${SYNTH[@]}" "${SHOLD[@]}" \
    --max-clips 100 --scores-tag t0-densepolicy-synth
step "T0-3 dense_above policy — real"
$PY scripts/eval.py --ckpt $CKPT "${REAL[@]}" "${RHOLD[@]}" \
    --max-clips 100 --scores-tag t0-densepolicy-real

# ---- T0-4: scale drift over 4x longer clips ----
step "T0-4 scale drift — 32-frame clips, synthetic"
$PY scripts/eval.py --ckpt $CKPT "${SYNTH[@]}" "${SHOLD[@]}" \
    --clip-len 32 --max-clips 30 --scores-tag t0-drift32-synth
step "T0-4 scale drift — 32-frame clips, real"
$PY scripts/eval.py --ckpt $CKPT "${REAL[@]}" "${RHOLD[@]}" \
    --clip-len 32 --max-clips 30 --scores-tag t0-drift32-real

# ---- T0-1: baselines on the same per-source protocol (100 clips/source) ----
for SRC in "vkitti2:$D/vkitti2::Scene06" \
           "tartanair2:$D/tartanair_v2::OldTownFall" \
           "pointodyssey:$D/pointodyssey::/pointodyssey/val/,/pointodyssey/test/" \
           "tum:$D/tum_static::walking_static" \
           "bonn:$D/bonn/rgbd_bonn_dataset::rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far"; do
    SPEC=${SRC%%::*}; HOLD=${SRC##*::}; NAME=${SPEC%%:*}
    for B in da2 da3; do
        step "T0-1 baseline $B — $NAME"
        EVAL_SPECS="$SPEC" EVAL_HOLDOUT="$HOLD" EVAL_TAG=" [$NAME]" MAX_CLIPS=100 \
            /home/hyunsu/miniforge3/condabin/conda run -n baselines --no-capture-output \
            python scripts/eval_baseline_$B.py
    done
done

# ---- T0-2: seed repeats, bin CE on/off, 8k steps each from the v8 init ----
for SEED in 0 1 2; do
    for BW in 0.2 0.0; do
        NAME=seed$SEED-bin$BW
        step "T0-2 $NAME"
        $PY scripts/train.py --config configs/main_v8.toml \
            --resume work_dirs/main_v8/latest.pt --resume-partial \
            --steps 8000 --seed "$SEED" --bin-weight "$BW" \
            --work-dir "work_dirs/t0-$NAME"
        $PY scripts/eval.py --ckpt "work_dirs/t0-$NAME/latest.pt" \
            "${REAL[@]}" "${RHOLD[@]}" --max-clips 100 --scores-tag real
        $PY scripts/eval.py --ckpt "work_dirs/t0-$NAME/latest.pt" \
            "${SYNTH[@]}" "${SHOLD[@]}" --max-clips 100 --scores-tag synth
    done
done

step "TIER 0 QUEUE DONE"
