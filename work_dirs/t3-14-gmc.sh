#!/usr/bin/env bash
# T3-14: does Low-Res GMC actually absorb ego-motion on REAL footage?
# So far it was only shown on vkitti2 (synthetic, clean geometry). KITTI raw
# is the same scene type shot for real: rolling shutter, motion blur, exposure
# changes, moving traffic. RGB only -- this measures the active ratio, which is
# the claim; accuracy needs the GT download and is handled by tier3.sh.
#
# Pixel gating and feature gating live on different scales (MSE vs relative
# L1), so a single tau is not comparable across them. Sweeping both sides is
# the only honest comparison: what matters is the active ratio each reaches.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
RAW=/home/hyunsu/dataset_ssd/kitti_raw/2011_09_26

for DR in 2011_09_26_drive_0002_sync 2011_09_26_drive_0005_sync \
          2011_09_26_drive_0013_sync 2011_09_26_drive_0020_sync; do
    D="$RAW/$DR/image_02/data"
    [ -d "$D" ] || continue
    echo; echo "======== $DR ========"
    for TAU in 0.02 0.05 0.1 0.2; do
        echo "-- pixel gating tau_on=$TAU"
        $PY scripts/infer.py --ckpt $CK --frames-dir "$D" --frames 60 \
            --no-gmc --tau-on $TAU --tau-off $(echo "$TAU/2" | bc -l) \
            --save-dir none 2>&1 | tail -2
    done
    for TAU in 0.05 0.1 0.2 0.4; do
        echo "-- GMC + feature gating tau_on=$TAU"
        $PY scripts/infer.py --ckpt $CK --frames-dir "$D" --frames 60 \
            --gmc --tau-on $TAU --tau-off $(echo "$TAU/2" | bc -l) \
            --save-dir none 2>&1 | tail -2
    done
done
echo; echo "T3-14 DONE"; date -Is
