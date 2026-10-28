#!/bin/bash
# Continuous storage sweep (owner, 2026-10-28: keep freeing disk space all the time).
# Run by the natlang-storage-sweep user timer every 30 minutes. Deletes only disposable scratch that no process
# has open and that has not been modified for SWEEP_AGE_MINUTES: test temp roots, natlang scratch directories in
# /tmp, torch inductor caches and the external test scratch. Logs what it removed and the free space afterwards;
# writes a low-space warning when root falls below SWEEP_ALERT_GB.
set -uo pipefail
AGE=${SWEEP_AGE_MINUTES:-180}
ALERT_GB=${SWEEP_ALERT_GB:-60}
LOG=${SWEEP_LOG:-$HOME/.local/state/natlang/storage-sweep.log}
mkdir -p "$(dirname "$LOG")"
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

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
echo "$(date -Is) sweep removed $((removed / 1073741824)) GiB; root free ${free_gb} GiB" >> "$LOG"
if [ "$free_gb" -lt "$ALERT_GB" ]; then
  echo "$(date -Is) LOW ROOT SPACE: ${free_gb} GiB free (< ${ALERT_GB})" >> "$LOG"
  echo "{\"at\": $(date +%s), \"root_free_gb\": $free_gb}" > "$HOME/.local/state/natlang/storage-low.json"
else
  rm -f "$HOME/.local/state/natlang/storage-low.json"
fi
