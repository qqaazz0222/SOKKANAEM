#!/usr/bin/env bash
# Serial GPU queue: wait out the capacity probe, then run the D1 arms.
set -u
while pgrep -f "probe_capacity.sh" > /dev/null; do sleep 60; done
exec bash scripts/d1_arms.sh
