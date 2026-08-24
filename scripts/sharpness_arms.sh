#!/usr/bin/env bash
# Three one-change arms against the blur in Figure 9. Serialized -- one GPU.
#
# Diagnosis first (scripts/sharpness_probe.py, scripts/sharp_metric.py):
#   * the patch grid is not the bottleneck. Ground truth pushed through our own
#     16px token grid keeps the walking person's silhouette and scores 0.0379
#     AbsRel on Bonn against our 0.1589.
#   * 512px inference is worse, so input resolution is not a free win either.
#   * boundary gradient ratio, the number that actually sees the blur: ours
#     0.43, DA v2 Small 0.76. AbsRel does not see it at all -- ours is the
#     better number on the static indoor scene while looking like a blob.
#
# Each arm mirrors the reported recipe exactly (base -> long-clip -> 8k stage,
# clip 24, batch 2, warp 2.0, edge 2.0, spread 0.5, seed 0) and changes one
# thing, so every row is comparable to work_dirs/v11-longclip-spread-s0.
#
#   A  fuse_norm         the DPT decoder's RGB skips are 5-10x too quiet to
#                        reach the output; zeroing the stem entirely moves
#                        AbsRel 1.6%. One GroupNorm per fusion branch.
#   B  --loss-space log  grad_loss and normal_loss took raw metres, where the
#                        same shape defect scores 74x louder at 10x the
#                        distance, and multiscale_grad_loss normalized
#                        disparity per BATCH, which down-weights a narrow
#                        indoor frame by up to 14x against its batch-mates.
#   C  teacher-grad      distil DA v2's disparity GRADIENTS. Distilling its
#                        values was measured to hurt (main_v8.toml); gradients
#                        carry the sharpness without pinning the depth range,
#                        which was the failure that removed the value term.
#
# Arm A is launched separately and may already be running; this waits for it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
BASE=work_dirs/v10-longclip/latest.pt
SHARP=outputs/sharpness/scores.jsonl
REAL=(--data "tum:$D/tum_static" --data "bonn:$D/bonn/rgbd_bonn_dataset")
RHOLD=(--holdout walking_static --holdout rgbd_bonn_crowd2
       --holdout rgbd_bonn_person_tracking2 --holdout rgbd_bonn_static_close_far)
COMMON=(--resume "$BASE" --resume-partial --clip-len 24 --batch 2 --steps 8000
        --seed 0 --warp-weight 2.0 --edge-weight 2.0 --spread-weight 0.5)

wait_for_gpu() {
    while pgrep -f "python.*scripts/train\.py" > /dev/null; do
        echo "[$(date -Is)] waiting for a training run..."; sleep 120
    done
}

score() {   # score <work_dir> <label>
    local W=$1 LABEL=$2
    $PY scripts/sharp_metric.py --ckpt "$W/latest.pt" --label "$LABEL" \
        --out "$SHARP" 2>&1 | tail -6
    $PY scripts/eval.py --ckpt "$W/latest.pt" "${REAL[@]}" "${RHOLD[@]}" \
        --clip-len 8 --max-clips 100 --scores-tag sharparm 2>&1 | tail -12
}

arm() {     # arm <work_dir> <label> <extra flags...>
    local W=$1 LABEL=$2; shift 2
    wait_for_gpu
    if [ -f "$W/latest.pt" ]; then
        echo "[$(date -Is)] $LABEL already trained, scoring only"
    else
        echo "[$(date -Is)] === $LABEL ==="
        $PY scripts/train.py "${COMMON[@]}" "$@" --work-dir "$W" \
            > "$W.log" 2>&1 || { echo "$LABEL FAILED, see $W.log"; return 1; }
    fi
    score "$W" "$LABEL"
}

# A is launched by hand ahead of this script; wait for it and score it.
wait_for_gpu
[ -f work_dirs/v12a-fusenorm-s0/latest.pt ] && \
    score work_dirs/v12a-fusenorm-s0 "A fuse_norm"

arm work_dirs/v12b-lossspace-s0 "B loss-space log" \
    --config configs/main_v8.toml --loss-space log

arm work_dirs/v12c-teachergrad-s0 "C teacher-grad 0.5" \
    --config configs/main_v8.toml --teacher-grad-weight 0.5

echo "[$(date -Is)] all arms done"
$PY - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("outputs/sharpness/scores.jsonl")]
k = ("absrel", "absrel_edge", "grad_ratio")
print(f"\n{'arm':<26s}" + "".join(f"{x:>13s}" for x in k))
for r in rows:
    b = r["scores"]["balanced"]
    print(f"{r['label']:<26s}" + "".join(f"{b[x]:>13.4f}" for x in k))
PYEOF
