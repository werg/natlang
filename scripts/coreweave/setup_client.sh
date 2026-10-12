#!/bin/bash
# Create the CoreWeave client venv used by scripts/coreweave/cwctl.py and bundle.py on DGX/Pop:
# the official cwsandbox client with its CLI (>= 1.14.2 for GPU requests, docs gpu-sandboxes.md) and boto3 >= 1.42.5
# (older boto3 refuses the non-loopback container-credentials host api.coreweave.com, docs object-storage-access.md).
# Installs nothing system-wide and needs no credentials. Re-run to upgrade.
set -euo pipefail
VENV=${NATLANG_CW_VENV:-$HOME/.local/share/natlang/cw-venv}
command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
uv pip install -q -p "$VENV" --upgrade 'cwsandbox[cli]>=1.14.2' 'boto3>=1.42.5'
"$VENV/bin/python" -c "import cwsandbox, boto3; print({'venv': '$VENV', 'cwsandbox': cwsandbox.__version__, 'boto3': boto3.__version__})"
