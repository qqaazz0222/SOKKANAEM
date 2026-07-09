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
    # TartanAir V2, direct from HF resolve URLs + wget -c — NOT the official
    # pypi package. That package wraps huggingface_hub's snapshot_download,
    # which on this network kept abandoning stalled multi-GB downloads and
    # restarting from byte 0 under a NEW randomly-suffixed .incomplete file
    # instead of resuming the existing one (confirmed: multiple same-content
    # .incomplete files per blob, hours apart, none growing). wget -c against
    # the plain resolve URL (HF redirects to a signed, Range-capable CDN
    # URL) was verified to resume correctly across a kill mid-transfer.
    #
    # 10 diverse envs, front cam, image+depth, easy — ~0.3M-frame scale of
    # the VDA training mix. Add envs to scale up (73 available upstream).
    need_space_gb 400
    local repo=https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main
    local envs=(AbandonedFactory AmusementPark Hospital JapaneseAlley
                ModularNeighborhood Office OldTownFall SeasonalForestSpring
                Downtown Supermarket)
    local out=$ROOT/tartanair_v2
    local zips=$out/_zips
    mkdir -p "$zips"
    for env in "${envs[@]}"; do
        for mod in image depth; do
            # already extracted for this env+modality? skip.
            if find "$out/$env/Data_easy" -mindepth 2 -maxdepth 2 \
                    -type d -name "${mod}_lcam_front" -print -quit \
                    2>/dev/null | grep -q .; then
                echo "== $env/$mod already extracted, skip"
                continue
            fi
            local zip=$zips/${env}_${mod}.zip
            local url=$repo/$env/Data_easy/${mod}_lcam_front.zip
            echo "== $url"
            # --timeout: wget has no read-timeout by default, so a
            # connection that goes silent without closing (the CLOSE-WAIT
            # stall we hit with the hf_hub-based downloader) would hang
            # forever; this makes wget itself detect and retry it.
            wget -c -q --show-progress --tries=0 --retry-connrefused \
                --timeout=60 --waitretry=5 -O "$zip" "$url" \
                || { echo "FAILED (will retry on next run): $url" >&2; continue; }
            mkdir -p "$out/$env"
            unzip -n -q "$zip" -d "$out/$env" && rm -f "$zip"
        done
    done
    echo "tartanair_v2 done -> $out"
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
