#!/bin/bash
# /mnt/external cleanup 2026-11-23: the two bf95-lowlr-1105 trees the 2026-11-19 pass (delete-20261119-external.sh)
# missed. The lineage was retired 2026-11-12 (no w5 checkpoint beat the base; DECISIONS "bf95-lowlr-1105 lineage").
# Checked before deletion: no process (cwd, exe, fds, maps, cmdline, environ), active or timer-triggered unit, ops watch
# script or data-registry record names either path; the sft-v7/v8/v9 checkpoints' pickled strings name neither.
# A path is kept when the reference check errors or reports a reference.
set -u
PATHS=(
  /mnt/external/dgx-moved-20261104/bf95-lowlr-1105     # 147 GB: w0-w5 learner/round trees, rl-snapshots
  /mnt/external/dgx-ops-archive-20261118/bf95-lowlr-1105  # 57 GB: controller, core-focus, heldeval, w5 stores
)
for path in "${PATHS[@]}"; do
  if ! /usr/bin/python3 /home/werg/natlang/scripts/prune_stale_checkouts.py --check "$path" >/dev/null 2>&1; then
    echo "kept (referenced or check failed): $path"; continue
  fi
  bytes=$(du -sxb -- "$path" 2>/dev/null | cut -f1)
  rm -rf -- "$path" && echo "deleted $path ${bytes:-?} bytes"
done
