#!/usr/bin/env bash
# T2-10 re-measurement: the fused Triton scan replaces the chunked pairwise
# PyTorch scan (forward only). Same six active ratios as REPORT 4.22 so the
# tables are directly comparable.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
CK=work_dirs/v9-60k/latest.pt
A="0.05 0.1 0.22 0.32 0.5 0.7"

for spec in "dense eager:--cache off" \
            "dense compiled:--cache off --compile" \
            "sparse eager:--cache on" \
            "sparse bucket eager:--cache on --bucket 64" \
            "sparse bucket compiled:--cache on --bucket 64 --compile" \
            "sparse bucket compiled fp16:--cache on --bucket 64 --compile --half"; do
    name=${spec%%:*}; flags=${spec#*:}
    echo; echo "################ $name ################"; date -Is
    $PY scripts/bench.py --ckpt $CK $flags --active $A
done
echo; echo "################ 4 streams ################"
$PY scripts/bench.py --ckpt $CK --cache on --bucket 64 --compile --streams 4 --active 0.22
$PY scripts/bench.py --ckpt $CK --cache off --compile --streams 4 --active 0.22
echo "T2-10 BENCH DONE"; date -Is
