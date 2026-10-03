#!/usr/bin/env bash
# Continue an interrupted S1 conversion without replacing its published or pending files.
# Usage: neuralese_resume_conversion.sh BUILD_DIR PROTECTED_JSON [PYTHON] [FRESH_TAG]
set -euo pipefail
BUILD=${1:?Original build directory is required}
PROTECTED_INPUT=${2:?Protected-source snapshot is required}
PY=${3:-/home/werg/bgkit/.venv/bin/python}
TAG=${4:-resume-$(date -u +%Y%m%dT%H%M%S%N)}
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid fresh tag" >&2; exit 2; }
[[ -d "$BUILD/raw" && -f "$PROTECTED_INPUT" ]] || { echo "Missing original raw data or protected snapshot" >&2; exit 2; }
TOOL=$(realpath "$(dirname "$0")/neuralese_port_records.py")
RESUME="$BUILD/$TAG"
mkdir "$RESUME"
mkdir "$RESUME/raw"
cp --no-clobber "$PROTECTED_INPUT" "$RESUME/protected.json"
exec > >(tee -a "$RESUME/build.log") 2>&1
run() { echo "== $(date -Is) $*"; ionice -c 3 nice -n 15 "$PY" "$TOOL" "$@"; }

# This entry point resumes the documented full-20261003 boundary, not arbitrary builds.
# Refuse to silently duplicate a SWE/agent stage that has already completed.
"$PY" - "$BUILD" "$RESUME" "$TOOL" <<'PY'
import hashlib, json, sys
from pathlib import Path
build, resume, tool = map(Path, sys.argv[1:])
for name in ("bgkit", "schnitzeljagd", "schnitzeljagd-turns"):
    summary = build / "raw" / f"{name}.summary.json"
    entries = json.loads(summary.read_text())
    if not entries:
        raise SystemExit(f"Empty completed-stage summary: {summary}")
    for entry in entries:
        for suffix in ("port-records", "rejects"):
            path = build / "raw" / f"{entry['name']}.{suffix}.jsonl"
            if not path.is_file():
                raise SystemExit(f"Missing completed-stage artifact: {path}")
if (build / "raw" / "swe.summary.json").exists() or (build / "raw" / "agents.summary.json").exists():
    raise SystemExit("SWE/agents already completed; review their receipts before choosing a new boundary")
files = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
         for p in sorted((build / "raw").iterdir()) if p.is_file()]
receipt = {"schema": "natlang.neuralese-conversion-resume/1", "original": str(build),
           "resume": str(resume), "original_artifacts": files,
           "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
           "protected_sha256": hashlib.sha256((resume / "protected.json").read_bytes()).hexdigest(),
           "policy": "Preserve original files; recompute unpublished SWE stage in a fresh directory",
           "publication": "candidate_only_pending_quality_review"}
(resume / "resume.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
run convert-swe --windows 2 --out "$RESUME/raw"
run convert-agents --out "$RESUME/raw"
run finalize "$BUILD/raw" "$RESUME/raw" --protected "$RESUME/protected.json" --out "$RESUME/final"
echo "Conversion candidate complete; review manifest before admitting data: $RESUME/final"
