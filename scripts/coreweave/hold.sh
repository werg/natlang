#!/bin/bash
# Keep a sandbox alive for exec'ed work (COREWEAVE_SANDBOXES.md: long-lived sandboxes, several jobs exec'ed into one).
# Pushing inputs from DGX is slow (measured 0.2-0.5 MB/s on 2026-10-28), so a pushed sandbox is reused rather than
# relaunched per attempt. Exits when /work/release exists or after HOLD_HOURS (default: the job lifetime decides).
#
#   hold.sh [HOLD_HOURS]
# Exec'ed jobs need the bootstrap environment: `source /work/.hold-env` (written here) before running.
set -u
{ echo "export PYTHONPATH='$PYTHONPATH'"; echo "export PATH='$PATH'"; } > /work/.hold-env
end=$(( $(date +%s) + ${1:-720} * 3600 ))
while [ ! -f /work/release ] && [ "$(date +%s)" -lt "$end" ]; do sleep 20; done
