#!/bin/sh
# Wait while the filesystem holding PATH has less than MIN_GB free (NVMe guard for data generators; coordinator
# 2026-10-12: generators pause below ~100 GB free and resume when space is back). Returns at once when there is room.
#   scripts/disk_guard.sh [PATH] [MIN_GB]       defaults: $HOME/data, $NATLANG_DISK_MIN_GB or 100
# Shell drivers call it before each unit of work; Python drivers use the same rule (static_curriculum_campaign.py).
path=${1:-$HOME/data}; min=${2:-${NATLANG_DISK_MIN_GB:-100}}
said=0
# Also pause while a trainer holds an io-quiet marker (checkpoint write in progress; natlang_neuralese/common/io_quiet.py).
quiet=$(dirname "$0")/../training/neuralese/natlang_neuralese/common/io_quiet.py
while :; do
  if reason=$(python3 "$quiet" 2>/dev/null); then
    [ $said = 0 ] && echo "disk guard: io-quiet marker live ($reason), pausing" >&2
    said=1; sleep ${NATLANG_IO_QUIET_POLL_S:-10}; continue
  fi
  free=$(df -P -BG "$path" | awk 'NR==2 { sub("G", "", $4); print $4 }')
  [ "$free" -ge "$min" ] && { [ $said = 1 ] && echo "disk guard: ${free} GB free on $path, resuming" >&2; exit 0; }
  [ $said = 0 ] && echo "disk guard: ${free} GB free on $path < ${min} GB, pausing" >&2
  said=1
  sleep 120
done
