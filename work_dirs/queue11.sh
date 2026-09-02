#!/usr/bin/env bash
# Round 4 -- combine what round 3 separated.
#
# Round 3's two readings: (a) restricting the SHAPE terms to synthetic GT costs
# sharpness (0.431 -> 0.416) because our synthetic sources are outdoor/driving
# and the indoor boundaries then get no supervision at all -- DA V2 filled that
# hole with 62M pseudo-labels and we cannot; (b) trimming the real sensor's
# boundary band out of the METRIC terms is the best accuracy result so far
# (AbsRel 0.1253 -> 0.1215, near -2.8%). So: keep the trimming, drop the
# shape-source restriction, and push the one lever that moves sharpness with
# the ringing penalty that keeps it honest.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "queue10 done" work_dirs/queue10.log 2>/dev/null; do sleep 120; done

bash scripts/finetune_arm.sh r4a-b3-os2-trim-s0 --full-res \
  --boundary-weight 3.0 --overshoot-weight 2.0 --trim-real-band
bash scripts/finetune_arm.sh r4b-b6-os3-trim-s0 --full-res \
  --boundary-weight 6.0 --overshoot-weight 3.0 --trim-real-band

echo "=== round 4 table ==="
$PY scripts/sharp_metric.py --table outputs/sharpness/suite.jsonl
for a in r4a-b3-os2-trim-s0 r4b-b6-os3-trim-s0; do
  echo "=== gate: $a ==="
  $PY scripts/acc_gate.py "work_dirs/acc/$a-L8.json" \
    --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp "work_dirs/acc/$a-L8.json" \
    --sharp-label "$a" || true
done
echo "[$(date -Is)] queue11 done"
