#!/bin/bash
# Continuous storage sweep (owner, 2026-10-28: keep freeing disk space all the time).
# Run by the natlang-storage-sweep user timer every 30 minutes, under the memory ledger
# (scripts/systemd/natlang-storage-sweep.service). Deletes disposable scratch that no process has open and that has
# not been modified for SWEEP_AGE_MINUTES: test temp roots, natlang scratch directories in /tmp, torch inductor caches
# and the external test scratch. Then removes experiment pins and code worktrees on the root disk that no process,
# unit or watch/state file references and that are older than SWEEP_PIN_AGE_HOURS (prune_stale_checkouts.py;
# SWEEP_PINS=0 opts out), and moves cold expert-iteration rounds and collections to the HDD (ei_round_tier.py;
# SWEEP_EI=0 opts out). Logs what it removed and the free space afterwards; below a disk floor (root
# SWEEP_ROOT_FLOOR_GB, external SWEEP_EXTERNAL_FLOOR_GB) it appends a DISK FLOOR line to the ops log
# (SWEEP_OPS_LOG) and the journal, and writes storage-low.json. The floors are checked first;
# SWEEP_ALERTS_ONLY=1 stops there (the unit's ExecStartPre, which needs no ledger admission). Every step is bounded
# (the pin pass by SWEEP_PIN_TIMEOUT, the whole run by the unit's timeout), so a slow disk cannot wedge the timer.
set -uo pipefail
AGE=${SWEEP_AGE_MINUTES:-180}
ALERT_GB=${SWEEP_ALERT_GB:-60}
ROOT_FLOOR_GB=${SWEEP_ROOT_FLOOR_GB:-40}
EXTERNAL_FLOOR_GB=${SWEEP_EXTERNAL_FLOOR_GB:-50}
PIN_AGE_HOURS=${SWEEP_PIN_AGE_HOURS:-6}
OPS_LOG=${SWEEP_OPS_LOG:-$HOME/data/lfm3-rl-dgx-20261029/ops/storage-alerts.log}
LOG=${SWEEP_LOG:-$HOME/.local/state/natlang/storage-sweep.log}
mkdir -p "$(dirname "$LOG")"
# Disk floors: one line per sweep while a disk is under its floor, in the ops log and the journal.
floor_alert() {  # MOUNT FLOOR_GB
  mountpoint -q "$1" || [ "$1" = / ] || return 0
  local avail
  avail=$(df -BG --output=avail "$1" | tail -1 | tr -dc 0-9)
  if [ -n "$avail" ] && [ "$avail" -lt "$2" ]; then
    local line="$(date -Is) DISK FLOOR $1: ${avail} GiB free < floor $2 GiB (storage_sweep; free space: AGENTS.md Storage hygiene)"
    echo "$line" >> "$LOG"
    mkdir -p "$(dirname "$OPS_LOG")" 2>/dev/null && echo "$line" >> "$OPS_LOG"
    echo "$line" >&2
  fi
}
floor_alert / "$ROOT_FLOOR_GB"
floor_alert /mnt/external "$EXTERNAL_FLOOR_GB"
# the unit's ExecStartPre: the alerts alone, which need no memory, so they fire even when the ledger cannot admit
[ "${SWEEP_ALERTS_ONLY:-0}" = 1 ] && exit 0
# A live pytest root must not prevent pruning its completed sibling runs. Never
# delete the whole pytest-of directory: its pytest-current and live PID locks
# belong to ongoing tests even when they have no files open at this instant.
for root in /tmp/pytest-of-*; do
  [ -d "$root" ] || continue
  /usr/bin/python3 "$(dirname "$0")/prune_pytest_scratch.py" --root "$root" --age-minutes "$AGE" --apply >> "$LOG" 2>&1
done
open_paths=$(mktemp)
trap 'rm -f "$open_paths"' EXIT
/usr/bin/python3 - "$(cd "$(dirname "$0")/.." && pwd)" > "$open_paths" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from scripts.archive_to_external import open_paths, process_cwds
for path in sorted(open_paths() | process_cwds()):
    if path.startswith(('/tmp/', '/mnt/external/')):
        print(path)
PY

candidates() {
  find /tmp -maxdepth 1 -mindepth 1 -user "$USER" \( -name 'natlang-*' -o -name 'torchinductor_*' -o -name 'tmp*' \) -mmin +"$AGE"
  [ -d /mnt/external/natlang-test-tmp ] && find /mnt/external/natlang-test-tmp -maxdepth 1 -mindepth 1 -user "$USER" -mmin +"$AGE"
}

removed=0
while IFS= read -r path; do
  # A directory is in use when any open file or working directory lies beneath it.
  if grep -qF -- "$path" "$open_paths"; then continue; fi
  # Only the top-level mtime is old; skip trees that something wrote to recently.
  if [ -d "$path" ] && [ -n "$(find "$path" -newermt "-$AGE minutes" -print -quit 2>/dev/null)" ]; then continue; fi
  bytes=$(du -sb -- "$path" 2>/dev/null | cut -f1)
  chmod -R u+w -- "$path" 2>/dev/null
  if rm -rf -- "$path" 2>/dev/null; then
    removed=$((removed + ${bytes:-0}))
    echo "$(date -Is) removed $path ${bytes:-0}" >> "$LOG"
  fi
done < <(candidates)

# Expert-iteration rounds: collections of policies the builds no longer read and the bulk of rounds older than the
# previous one move to the HDD (verified, symlinked back); regenerable exports go (scripts/ei_round_tier.py).
# SWEEP_EI=0 opts out; bounded like every step.
if [ "${SWEEP_EI:-1}" != 0 ]; then
  timeout -k 30 "${SWEEP_EI_TIMEOUT:-1200}" /usr/bin/python3 "$(dirname "$0")/ei_round_tier.py" --apply \
    --max-minutes "${SWEEP_EI_MINUTES:-18}" > /dev/null 2>&1
  [ $? = 124 ] && echo "$(date -Is) ei_round_tier timed out after ${SWEEP_EI_TIMEOUT:-1200} s" >> "$LOG"
fi

if [ "${SWEEP_PINS:-1}" != 0 ]; then
  # bounded: a pass that cannot finish in SWEEP_PIN_TIMEOUT seconds is killed and logged; the next sweep retries
  pins_out=$(mktemp)
  timeout -k 30 "${SWEEP_PIN_TIMEOUT:-900}" /usr/bin/python3 "$(dirname "$0")/prune_stale_checkouts.py" --apply \
    --min-age-hours "$PIN_AGE_HOURS" > "$pins_out" 2>&1
  [ $? = 124 ] && echo "$(date -Is) prune_stale_checkouts timed out after ${SWEEP_PIN_TIMEOUT:-900} s" >> "$LOG"
  grep -v '"kept"' "$pins_out" >> "$LOG"
  pin_bytes=$(sed -n 's/.*"summary": true.*"freed_bytes": \([0-9]*\).*/\1/p' "$pins_out" | tail -1)
  removed=$((removed + ${pin_bytes:-0}))
  rm -f "$pins_out"
fi

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
echo "$(date -Is) sweep removed $((removed / 1073741824)) GiB; root free ${free_gb} GiB" >> "$LOG"
# the target the tiers keep the root disk at when they can supply it (the floor above is the alert line)
if [ "$free_gb" -lt "${SWEEP_ROOT_TARGET_GB:-60}" ]; then
  echo "$(date -Is) ROOT BELOW TARGET: ${free_gb} GiB free < ${SWEEP_ROOT_TARGET_GB:-60} after the tiers (nothing more movable)" >> "$LOG"
fi
if [ "$free_gb" -lt "$ALERT_GB" ]; then
  echo "$(date -Is) LOW ROOT SPACE: ${free_gb} GiB free (< ${ALERT_GB})" >> "$LOG"
  echo "{\"at\": $(date +%s), \"root_free_gb\": $free_gb}" > "$HOME/.local/state/natlang/storage-low.json"
else
  rm -f "$HOME/.local/state/natlang/storage-low.json"
fi
