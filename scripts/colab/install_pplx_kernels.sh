#!/usr/bin/env bash
# Install differentiable FLA / causal-conv1d / Liger kernels in the Colab A100 venv so T2's
# PplxWarmstartWriter runs 3.6x faster (commit b76f59d2f) instead of transformers' slow reference PyTorch path.
set -euo pipefail

PYTHON="${1:-/content/natlang-work/natlang/.venv-neuralese/bin/python}"
PIP_CMD=("$PYTHON" -m pip)

echo "[install_pplx_kernels] Python: $PYTHON"
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# Install prebuilt/no-build-isolation wheels matching the active torch/CUDA ABI
"${PIP_CMD[@]}" install --no-build-isolation --no-deps "flash-linear-attention>=0.2.0" "causal-conv1d>=1.4.0" "liger-kernel>=0.5.0"

"$PYTHON" - <<'PY'
import fla, causal_conv1d
print("[install_pplx_kernels] OK: fla =", getattr(fla, "__version__", "installed"),
      "causal_conv1d =", getattr(causal_conv1d, "__version__", "installed"))
PY
