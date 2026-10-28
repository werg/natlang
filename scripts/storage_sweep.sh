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
open_paths=$(mktemp)
trap 'rm -f "$open_paths"' EXIT
for p in /proc/[0-9]*; do
  readlink "$p/cwd" 2>/dev/null
  ls -l "$p/fd" 2>/dev/null | awk '{print $NF}'
done | grep -E '^(/tmp|/mnt/external)/' | sort -u > "$open_paths"

candidates() {
  find /tmp -maxdepth 1 -mindepth 1 -user "$USER" \( -name 'pytest-of-*' -o -name 'natlang-*' -o -name 'torchinductor_*' -o -name 'tmp*' \) -mmin +"$AGE"
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
