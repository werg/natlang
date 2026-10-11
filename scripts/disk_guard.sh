#!/bin/sh
# Wait while the filesystem holding PATH has less than MIN_GB free (NVMe guard for data generators; coordinator
# 2026-10-12: generators pause below ~100 GB free and resume when space is back). Returns at once when there is room.
#   scripts/disk_guard.sh [PATH] [MIN_GB]       defaults: $HOME/data, $NATLANG_DISK_MIN_GB or 100
# Shell drivers call it before each unit of work; Python drivers use the same rule (static_curriculum_campaign.py).
path=${1:-$HOME/data}; min=${2:-${NATLANG_DISK_MIN_GB:-100}}
said=0
while :; do
  free=$(df -P -BG "$path" | awk 'NR==2 { sub("G", "", $4); print $4 }')
  [ "$free" -ge "$min" ] && { [ $said = 1 ] && echo "disk guard: ${free} GB free on $path, resuming" >&2; exit 0; }
  [ $said = 0 ] && echo "disk guard: ${free} GB free on $path < ${min} GB, pausing" >&2
  said=1
  sleep 120
done
