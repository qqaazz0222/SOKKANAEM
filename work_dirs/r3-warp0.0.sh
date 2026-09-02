#!/usr/bin/env bash
# Extends Table 14's warp-weight dial to the low side. The paper only swept
# up (2.0 reported -> 4.0 -> 8.0) and found accuracy monotonically WORSENS as
# the weight rises; Section 5.9/6.4 suspect the flow-warp term as part of why
# the model sits 3.5x above its own ceiling on the dynamic-object source
# (Bonn), since the term assumes photometric correspondence that breaks at a
# moving object's depth discontinuity. This arm asks the converse question:
# does removing the term (warp_weight=0, which skips computing it entirely --
# see scripts/train.py:406) buy back accuracy, and at what cost to the
# temporal metrics it was added for?
#
# Exact same protocol as the existing warp4.0/warp8.0 arms (work_dirs/v11-
# warp4.0, .../v11-warp8.0): single 8k-step fine-tune from the long-clip
# checkpoint, everything else at the reported configuration, single seed.
# Needs the GPU to itself.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
WD=work_dirs/v11-warp0.0

echo "======== train: warp_weight=0.0 ========"; date -Is
$PY scripts/train.py --config configs/main_v8.toml \
    --resume work_dirs/v10-longclip/latest.pt --resume-partial \
    --clip-len 24 --batch 2 --steps 8000 --seed 0 \
    --warp-weight 0.0 --edge-weight 2.0 --spread-weight 0.5 \
    --work-dir "$WD"
echo "train done"; date -Is

HOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
      --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
DATA=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")

for L in 8 32 256; do
    for K in 30 60; do
        echo; echo "---- eval L$L K$K ----"; date -Is
        $PY scripts/eval.py --ckpt "$WD/latest.pt" "${DATA[@]}" "${HOLD[@]}" \
            --clip-len "$L" --size 256 --max-clips 100 --align median \
            --gate-mode delta --keyframe-every "$K" --dense-above 0.4 \
            --bin-temp 1.0 --scores-tag "warp0.0-L${L}K${K}" \
            | tee -a "$WD/eval.txt"
    done
done

echo; echo "---- range probe ----"; date -Is
$PY scripts/range_probe.py --ckpt "$WD/latest.pt" | tee "$WD/range.txt"

echo; echo "R3 WARP0.0 DONE"; date -Is
echo "then: $PY scripts/bootstrap_ci.py --ours $WD/scores_warp0.0-L256K30.json --suffix l256"
