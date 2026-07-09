#!/usr/bin/env bash
# Wraps download_data.sh with a stall watchdog: huggingface_hub's plain
# requests downloader occasionally lets a connection die silently
# (CLOSE-WAIT, no retry) and just hangs forever with zero throughput.
# This restarts the download (resumable via .incomplete files) whenever
# no byte has been written anywhere under $ROOT for STALL_SECS.
#
# Usage: scripts/download_watchdog.sh tartanair pointodyssey ...
set -uo pipefail

ROOT=/archive/Dataset_SOKKANAEM
STALL_SECS=600     # no growth in 10 min -> assume dead, restart
POLL_SECS=60
LOG=$ROOT/watchdog.log

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

newest_mtime() {
    # Any regular file's mtime, not just *.incomplete — wget writes
    # straight to the target zip (no special suffix), hf_hub uses
    # *.incomplete. Excluding logs keeps our own tee/log writes from
    # masking a stalled download.
    find "$ROOT" -type f ! -name "*.log" -printf '%T@\n' 2>/dev/null \
        | sort -rn | head -1
}

# Kill only actual descendants of $1 (never a process-group kill — that
# scope could reach the watchdog's own shell in a non-interactive script
# where backgrounded jobs share the parent's pgid).
kill_tree() {
    local p=$1
    local c
    for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
    kill -9 "$p" 2>/dev/null
}

run_one() {
    local ds=$1
    log "=== starting $ds ==="
    bash "$(dirname "$0")/download_data.sh" "$ds" \
        >> "$ROOT/${ds}_download.log" 2>&1 &
    local pid=$!
    local last_change=$(date +%s)
    local last_mtime=""
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_SECS"
        local mtime; mtime=$(newest_mtime)
        local now; now=$(date +%s)
        if [ "$mtime" != "$last_mtime" ]; then
            last_mtime=$mtime
            last_change=$now
        elif (( now - last_change > STALL_SECS )); then
            log "STALLED >${STALL_SECS}s (no file writes) — killing pid $pid tree and retrying"
            kill_tree "$pid"
            wait "$pid" 2>/dev/null
            run_one "$ds"   # resumes from .incomplete, recurse replaces this attempt
            return
        fi
    done
    wait "$pid"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        log "$ds exited rc=$rc — retrying"
        run_one "$ds"
        return
    fi
    log "=== $ds done ==="
}

for ds in "$@"; do
    run_one "$ds"
done
log "=== watchdog finished all: $* ==="
