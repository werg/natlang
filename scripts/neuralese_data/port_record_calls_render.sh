#!/bin/bash
# Render the port-record call corpora (D15/D16) for training, in order, once each is published: S3 v1 (frozen at
# 2,000-character pages), S3 v2, view-ask v2, s3-large v2. Each corpus goes through the shared parts renderer
# (render_parts.py: parts of ~12k decisions at program boundaries, one part decompressed at a time; program-header and
# whole-trajectory for every program, full only for the 5% warm-up sample) and registers itself as <id>-render
# through register_corpora.py (no hand commit). Resumable: finished parts and registered renders are skipped.
# Run as a light waiter (part renders are their own low-priority ledger units):
#   systemd-run --user --unit s1-natlang-render --working-directory=/home/werg/natlang \
#     scripts/neuralese_data/port_record_calls_render.sh
set -uo pipefail
cd /home/werg/natlang
DATA=$HOME/data/natlang-corpora
render() { # corpus-id
  local id=$1 out=$DATA/$1-render
  [ -f "$out/registered" ] && return 0
  until [ -f "$DATA/$id/published" ]; do sleep 600; done
  local tries=0
  until python3 scripts/neuralese_data/render_parts.py render --corpus-id "$id" --out "$out" \
      --source $(ls "$DATA/$id"/turns/*/*.turns.jsonl.zst | sort) --unit-prefix "render-$id" \
      --register "data/neuralese/corpora/$id-render" --link >> "$DATA/$id-render.log" 2>&1; do
    tries=$((tries + 1)); echo "$(date -Is) $id render attempt $tries failed" >&2
    [ $tries -ge 6 ] && return 1
    sleep 900
  done
  touch "$out/registered"
  # Raw outputs move to the HDD once their render is registered (coordinator 2026-10-12): left to the archiver
  # (scripts/archive_to_external.py), which updates registry paths; not done here.
}
render port-record-calls-s3-20261011-v1
render port-record-calls-s3-20261011-v2
render port-record-calls-view-ask-20261011-v2
render port-record-calls-s3-large-20261011-v2
