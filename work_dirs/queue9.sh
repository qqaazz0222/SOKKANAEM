#!/usr/bin/env bash
# Re-measure P2's sealed reference rows now that the sharpness block scores the
# ALIGNED depth (the first pass scored the raw prediction, which is a unit error
# for a relative baseline: DA2 read grad_ratio 1.24, DPT-Large edge AbsRel 6.15).
set -u
PY=/home/hyunsu/miniforge3/envs/sokkanaem/bin/python
until grep -q "queue8 done" work_dirs/queue8.log 2>/dev/null; do sleep 120; done
bash work_dirs/sharp_refs.sh
echo "=== sealed-manifest sharpness, every arm that has a dump ==="
$PY - <<'PY'
import json, pathlib
keys = ("absrel_edge","grad_ratio","boundary_f1","boundary_precision",
        "boundary_recall","flat_tv","overshoot")
print(f"{'arm':<26}" + "".join(f"{k:>19}" for k in keys))
for f in sorted(pathlib.Path("work_dirs/acc").glob("*-L8*.json")):
    d = json.loads(f.read_text())
    sh = d.get("sharp")
    if not sh:
        continue
    srcs = list(sh)
    bal = {k: sum(sh[s].get(k, float('nan')) for s in srcs)/len(srcs) for k in keys}
    print(f"{d['label'][:25]:<26}" + "".join(f"{bal[k]:>19.4f}" for k in keys))
PY
echo "[$(date -Is)] queue9 done"
