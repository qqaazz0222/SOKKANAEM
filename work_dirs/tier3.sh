#!/usr/bin/env bash
# Tier 3 — generalisation, run once the KITTI download lands.
#   setsid nohup bash work_dirs/tier3.sh > work_dirs/tier3-console.log 2>&1 < /dev/null &
#
# T3-13 cross-domain zero-shot: KITTI raw is real driving footage that was
# never trained on. Only its synthetic clone (vkitti2) was, so this measures
# synthetic->real transfer on the domain the model has the most exposure to
# in simulation and none in reality. NYU/ScanNet were not attempted: the NYU
# host times out and ScanNet needs a signed agreement.
#
# T3-14 GMC on real ego-motion: the same drives with --gmc on and off. The
# claim under test is that homography compensation absorbs camera-induced
# change so the mask reflects scene change -- so far only shown on synthetic
# clean geometry.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
ZS=/home/hyunsu/dataset_ssd/kitti_zs

echo "################ T3-13 zero-shot: KITTI raw, no GMC ################"; date -Is
$PY scripts/eval.py --ckpt $CK --data "kitti:$ZS" --max-clips 100 --scores-tag zs-kitti

echo; echo "################ T3-14 same drives, GMC on ################"; date -Is
$PY scripts/eval.py --ckpt $CK --data "kitti:$ZS" --max-clips 100 --gmc --scores-tag zs-kitti-gmc

echo; echo "################ T3-14b tau sweep with GMC ################"; date -Is
$PY scripts/eval.py --ckpt $CK --data "kitti:$ZS" --max-clips 30 --gmc --sweep-tau

echo; echo "################ in-domain reference: vkitti2 holdout ################"; date -Is
$PY scripts/eval.py --ckpt $CK --data vkitti2:/home/hyunsu/dataset_ssd/vkitti2 \
    --holdout Scene06 --max-clips 100 --scores-tag zs-vkitti-ref

echo "TIER 3 DONE"; date -Is
