#!/usr/bin/env bash
# r2 Major 1, the three baselines work_dirs/r2-scale.sh could not run: they
# live in their own conda envs (DA3 in `baselines`, VDA in `vda`) and DA-v2-
# Small was simply missing from that list. Same holdout, same clip lengths and
# the same native alignment rule each model is reported under in Table 3a.
set -uo pipefail
cd "$(dirname "$0")/.."
CONDA=/home/hyunsu/miniforge3/bin/conda
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
D=/home/hyunsu/dataset_ssd
export MAX_CLIPS=100

SRCS=("tum:$D/tum_static:walking_static"
      "bonn:$D/bonn/rgbd_bonn_dataset:rgbd_bonn_crowd2,rgbd_bonn_person_tracking2,rgbd_bonn_static_close_far")

for SRC in "${SRCS[@]}"; do
    NAME=${SRC%%:*}; REST=${SRC#*:}; PATHP=${REST%%:*}; HOLD=${REST#*:}
    for L in 8 256; do
        export CLIP_LEN=$L EVAL_SPECS="$NAME:$PATHP" EVAL_HOLDOUT="$HOLD" \
               EVAL_TAG=" $NAME l$L"

        echo; echo "---- DA-v2-Small / $NAME / L$L ----"; date -Is
        $PY scripts/eval_baseline_da2.py \
            --ckpt depth-anything/Depth-Anything-V2-Small-hf \
            --label "DA-v2-Small" --mode relative 2>/dev/null \
            | grep -E "pooled|holdout clips|per-clip"

        echo; echo "---- DA3-BASE / $NAME / L$L ----"; date -Is
        ALIGN=scaleshift $CONDA run --no-capture-output -n baselines \
            python scripts/eval_baseline_da3.py 2>/dev/null \
            | grep -E "pooled|holdout clips|per-clip"

        echo; echo "---- VDA / $NAME / L$L ----"; date -Is
        $CONDA run --no-capture-output -n vda \
            python scripts/eval_baseline_vda.py ~/checkouts/Video-Depth-Anything \
            2>/dev/null | grep -E "pooled|holdout clips|per-clip"
    done
done

echo; echo "R2 SCALE REST DONE"; date -Is
