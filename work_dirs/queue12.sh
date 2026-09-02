#!/usr/bin/env bash
set -u
until grep -q "bonn refs done" work_dirs/bonn-refs.log 2>/dev/null; do sleep 60; done
bash scripts/hires_arm.sh a1-384-s0
echo "[$(date -Is)] queue12 done"
