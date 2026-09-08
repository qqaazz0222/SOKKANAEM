#!/usr/bin/env bash
# Why the 60k distillation oversharpened, in two 8k arms.
#
# It minimised its loss (8.38 -> 5.11) and produced grad_ratio 1.80, flat TV
# 0.16, overshoot 0.76 and delta1 0.5427. The checkpoint says where it went:
# D1's full-resolution RGB branch grew to norm 4.84 against 1.95 in every
# fine-tuned arm, and up_mix to 1.68 against 0.17. With no pretrained shape
# prior, copying image edges is a cheaper way to satisfy a gradient-hungry loss
# than estimating depth -- albedo leaking into the depth map.
#
# a1 keeps D1 but stops pushing gradients: no teacher gradient term, boundary
#    down to 1, and the overshoot penalty on.
# a2 removes D1's full-res path entirely, everything else as a1, which is the
#    control that says whether the RGB branch is the leak or merely where it
#    showed up.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
T=work_dirs/q0-calib-s0/latest.pt

arm () {  # arm <name> <extra flags...>
  local name=$1; shift
  echo "[$(date -Is)] === $name : $* ==="
  $PY scripts/train.py --config configs/main_v8.toml \
    --dim 256 --depth 6 --d-state 24 \
    --steps 8000 --lr 3e-4 --warmup 500 \
    --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5 \
    --q-teacher "$T" --q-weight 1.0 --q-grad-weight 0.0 \
    --boundary-weight 1.0 --overshoot-weight 1.0 \
    --workers 8 --work-dir "work_dirs/$name" "$@"
  $PY scripts/sharp_metric.py --ckpt "work_dirs/$name/latest.pt" \
    --label "$name" --out outputs/sharpness/suite.jsonl
  $PY scripts/decide_next.py "$name" || true
  $PY - <<PYEOF
import torch
sd = torch.load("work_dirs/$name/latest.pt", map_location="cpu")
sd = sd.get("ema") or sd["model"]
ups = {k: float(v.norm()) for k, v in sd.items() if "decoder.up_" in k}
print("  D1 branch norms:", {k.split("decoder.")[1]: round(v, 3) for k, v in ups.items()} or "no D1")
PYEOF
}

arm m0-noleak-s0 --full-res
arm m0-nod1-s0
echo "[$(date -Is)] queue23 done"
