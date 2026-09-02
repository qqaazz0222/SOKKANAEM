#!/usr/bin/env bash
# Q1 retry, two conservative arms. The first attempt (shape-lr 1e-5, retention
# 1.0, boundary 0.5, everything unfrozen) regressed on every gauge and bought
# its gradient ratio with oversharpening -- 1.154 against GT, overshoot 0.347,
# edge AbsRel 0.187. The suite caught it, which is what it is for.
#
#   q1b: neck+head only. The DINOv2 encoder is what 142M images bought and
#        13k indoor clips are not an argument for moving it.
#   q1c: everything, but at a third the LR, four times the retention, and no
#        boundary term -- the term that did the oversharpening.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python

arm () {  # arm <name> <flags...>
  local name=$1; shift
  $PY scripts/train_q.py --steps 4000 --real-only --work-dir "work_dirs/$name" "$@"
  echo "[$(date -Is)] --- $name sharpness ---"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
    --label "$name" --out outputs/sharpness/suite.jsonl
  echo "[$(date -Is)] --- $name P1+P2 sealed L8 ---"
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
    --model "ours:work_dirs/$name/latest.pt" --regions --sharp --tag "$name-L8"
  $PY scripts/acc_gate.py "work_dirs/acc/$name-L8.json" \
    --sharp outputs/sharpness/suite.jsonl \
    --sharp work_dirs/acc/da2-small-L8-sharp.json \
    --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label "$name" || true
}

arm q1b-head-s0 --unfreeze-head --shape-lr 1e-5 --retention-weight 2.0
arm q1c-slow-s0 --unfreeze --shape-lr 3e-6 --retention-weight 4.0
echo "[$(date -Is)] queue15 done"
