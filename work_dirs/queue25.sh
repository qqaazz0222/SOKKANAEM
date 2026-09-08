#!/usr/bin/env bash
# Is the random-mask curriculum what turns the decoder into an edge detector?
#
# Both 60k runs oversharpened (grad_ratio 1.80 and 2.25) and both 8k probes did
# not (0.5624). The difference that tracks it is the skip ramp: max_skip 0.5 is
# reached at steps/2, so an 8k probe ends at skip 0.3 while a 60k run spends its
# second half at 0.5. With half the tokens masked, the only signal left at full
# resolution is D1's RGB stem, and the decoder leans on it -- which is exactly
# the leak the checkpoint norms showed.
#
# Two 8k arms at CONSTANT skip, everything else identical. If skip 0.5 alone
# reproduces the ratio blow-up in 8k steps, the mechanism is confirmed and the
# fix is to train dense (PLAN Stage A's own instruction) and reintroduce
# sparsity in Stage E rather than during the main run.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
T=work_dirs/q0-calib-s0/latest.pt

arm () {  # arm <name> <max-skip>
  local name=$1 skip=$2
  echo "[$(date -Is)] === $name : max-skip $skip ==="
  $PY scripts/train.py --config configs/main_v8.toml \
    --dim 256 --depth 6 --d-state 24 --full-res \
    --steps 8000 --lr 3e-4 --warmup 500 --max-skip "$skip" \
    --boundary-weight 1.0 --overshoot-weight 1.0 \
    --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
    --q-teacher "$T" --q-weight 1.0 --q-grad-weight 0.0 \
    --workers 8 --work-dir "work_dirs/$name"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
    --label "$name" --out outputs/sharpness/suite.jsonl
  $PY - <<PYEOF
import torch
sd = torch.load("work_dirs/$name/latest.pt", map_location="cpu")
sd = sd.get("ema") or sd["model"]
print("  up_stem norm:", round(float(sum(v.norm() for k, v in sd.items()
      if "up_stem" in k)), 3))
PYEOF
}

arm m0-skip50-s0 0.5     # the suspect, held at 0.5 from step 0
arm m0-dense-s0 0.0      # the control, and the recipe Stage A actually asks for
echo "[$(date -Is)] queue25 done"
