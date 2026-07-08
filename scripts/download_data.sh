#!/usr/bin/env bash
# Main-training dataset downloader -> /archive/Dataset_SOKKANAEM
#
# Mix follows Video Depth Anything (CVPR 2025): TartanAir + PointOdyssey
# (+ IRS, Dynamic Replica) alongside the vkitti2 already in data/.
#
# Usage:
#   scripts/download_data.sh tartanair      # ~8 envs, Easy, left cam (~250GB)
#   scripts/download_data.sh pointodyssey   # HF mirror (~100GB)
#   scripts/download_data.sh dynamic_replica # Meta links file (~500GB)
#   scripts/download_data.sh irs            # prints manual instructions
#   scripts/download_data.sh all
#
# All downloads are resumable (wget -c / hf resume). Re-run after failures.
# Run inside the sokkanaem conda env (python + pip needed):
#   conda run --no-capture-output -n sokkanaem scripts/download_data.sh tartanair
set -uo pipefail

ROOT=/archive/Dataset_SOKKANAEM
mkdir -p "$ROOT"

need_space_gb() { # fail early if less than $1 GB free
    local free
    free=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
    if (( free < $1 )); then
        echo "ERROR: need ${1}GB free on $ROOT, have ${free}GB" >&2
        exit 1
    fi
}

tartanair() {
    # TartanAir V2 via the official pypi package (v1 Azure blob is gone —
    # NXDOMAIN). HF-hosted, resumable. 10 diverse envs, front cam,
    # image+depth, easy — ~0.3M-frame scale of the VDA training mix.
    # Add envs to scale up (73 available: python -c "...ta.list_envs()").
    need_space_gb 400
    pip show tartanair >/dev/null 2>&1 || pip install -q tartanair
    python - <<'EOF'
import tartanair as ta

ta.init("/archive/Dataset_SOKKANAEM/tartanair_v2")
ta.download(
    env=["AbandonedFactory", "AmusementPark", "Hospital", "JapaneseAlley",
         "ModularNeighborhood", "Office", "OldTownFall",
         "SeasonalForestSpring", "Downtown", "Supermarket"],
    difficulty=["easy"],
    modality=["image", "depth"],
    camera_name=["lcam_front"],
    unzip=True, delete_zip=True, num_workers=4,
)
EOF
    echo "tartanair_v2 done -> $ROOT/tartanair_v2"
}

pointodyssey() {
    # Official release mirrored on HuggingFace. ~100GB.
    need_space_gb 150
    local out=$ROOT/pointodyssey
    command -v hf >/dev/null || pip install -q -U huggingface_hub
    hf download aharley/pointodyssey --repo-type dataset \
        --local-dir "$out" || {
        echo "FAILED: HF mirror unavailable — fall back to the official" >&2
        echo "Google Drive links at https://pointodyssey.com" >&2; return 1; }
    echo "pointodyssey done -> $out"
}

dynamic_replica() {
    # Meta 'dynamic_stereo' repo ships the official download tooling
    # (links file + downloader). CC-BY-NC. ~500GB full.
    need_space_gb 550
    local out=$ROOT/dynamic_replica
    mkdir -p "$out"
    if [ ! -d "$out/dynamic_stereo" ]; then
        git clone --depth 1 \
            https://github.com/facebookresearch/dynamic_stereo "$out/dynamic_stereo"
    fi
    echo "Run the official downloader (needs its own deps):"
    echo "  cd $out/dynamic_stereo && python ./scripts/download_dynamic_replica.py --target_folder $out"
    echo "(kept manual: Meta's links file / API changes occasionally)"
}

irs() {
    cat <<EOF
IRS (indoor stereo, dense GT) has no direct-download mirror — hosted on
OneDrive/BaiduYun via https://github.com/HKBU-HPML/IRS
Download 'IRS_small' (or full) manually into: $ROOT/irs
EOF
}

case "${1:-}" in
    tartanair)        tartanair ;;
    pointodyssey)     pointodyssey ;;
    dynamic_replica)  dynamic_replica ;;
    irs)              irs ;;
    all)              tartanair; pointodyssey; dynamic_replica; irs ;;
    *) echo "usage: $0 {tartanair|pointodyssey|dynamic_replica|irs|all}"; exit 1 ;;
esac
