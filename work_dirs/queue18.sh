#!/usr/bin/env bash
# (1) Q0 + streaming: the zero-init temporal adapter, then the long-stream
# protocol that P3 is written on. The adapter's job is stability, not per-frame
# accuracy -- §4.43 measured that recurrent state does not create the latter --
# so it trains with the warp (training-time TCE) term and is judged on L8 vs
# L256, where the frozen Q0 has no state at all and must drift like any
# stateless model.
set -eu
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "queue17 done" work_dirs/queue17.log 2>/dev/null; do sleep 120; done

$PY scripts/train_q.py --steps 4000 --clip-len 8 --batch 1 --real-only \
  --temporal --warp-weight 2.0 --work-dir work_dirs/qt-stream-s0
# clip 8 at 518px used to OOM: the temporal state now carries values, not the
# graph (truncated BPTT), and the peak dropped from >20 GiB to 1.6 GiB

$PY scripts/sharp_metric.py --ckpt work_dirs/qt-stream-s0/latest.pt \
  --label "qt-stream-s0" --out outputs/sharpness/suite.jsonl
for L in 8 256; do
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L$L.json \
    --model ours:work_dirs/qt-stream-s0/latest.pt --per-frame --sharp \
    --tag "qt-stream-s0-L$L-pf"
  # the frozen, stateless Q0 on the same protocol, as the drift control
  $PY scripts/eval_acc.py --manifest manifests/acc_real_L$L.json \
    --model ours:work_dirs/q0-calib-s0/latest.pt --per-frame \
    --tag "q0-calib-s0-L$L-pf"
done
echo "=== P3 (per-frame gauge) ==="
$PY scripts/acc_gate.py work_dirs/acc/qt-stream-s0-L8-pf.json \
  work_dirs/acc/qt-stream-s0-L256-pf.json --gauge scaleshift/frame \
  --sharp outputs/sharpness/suite.jsonl \
  --sharp work_dirs/acc/da2-small-L8-sharp.json \
  --sharp work_dirs/acc/ours-L8-sharp.json --sharp-label "qt-stream-s0" || true
$PY scripts/acc_gate.py work_dirs/acc/q0-calib-s0-L8-pf.json \
  work_dirs/acc/q0-calib-s0-L256-pf.json --gauge scaleshift/frame || true
echo "[$(date -Is)] queue18 done"
