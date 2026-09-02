#!/usr/bin/env bash
# The wrapper that was going to chain train -> eval -> range_probe (r3-warp0.0.sh)
# got killed, but the training subprocess it launched (pid passed as $1) was
# reparented to init and kept running unattended. This just waits for that
# orphan to finish, then does the eval + range_probe steps r3-warp0.0.sh never
# got to run.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
WD=work_dirs/v11-warp0.0
TRAIN_PID=${1:?usage: r3-warp0.0-continue.sh <orphaned train.py pid>}

echo "[$(date -Is)] waiting for train.py (pid $TRAIN_PID) to finish"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] train.py exited"
tail -5 "$WD/train.log"

if ! grep -q "saved -> $WD/latest.pt" "$WD/train.log"; then
    echo "[$(date -Is)] no final save line in $WD/train.log -- training did not" \
         "reach step 8000 cleanly. Not running eval on a possibly-crashed" \
         "checkpoint; leaving it for a human to look at."
    exit 1
fi

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
