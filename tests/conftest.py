import os
from pathlib import Path

# Runtime inventories, task collectors and exact source bindings in tests must
# describe the checkout under test. The logical ``repo`` root otherwise falls
# back to the DGX machine default, which does not exist on CI runners and is a
# different revision in a linked worktree. scripts/check-main.sh exports the
# same value; a plain ``python -m pytest`` run (the GitHub workflow) now gets it
# too. Tests of the default resolution remove it with monkeypatch.
os.environ["NATLANG_REPO"] = str(Path(__file__).resolve().parents[1])

# Native execution is batch invariant: a row's arithmetic must not depend on how
# many other rows share its GEMM (graph vs cached execution, chunked guidance
# replay, single- vs multi-token decode). CUDA uses the invariant kernels in
# natlang_neuralese.maple.fused_moe. On x86 CPU PyTorch's FP32 GEMM is MKL,
# whose default kernels take different code paths for different row counts
# (an M=1 product differs from the same row inside an M=5 product). MKL's strict
# conditional-numerical-reproducibility mode makes each output element's
# reduction independent of the matrix partitioning. MKL reads this setting at
# its first BLAS call, so it is set before any test imports torch, and child
# processes inherit it. It has no effect on non-MKL builds (aarch64/DGX).
os.environ.setdefault("MKL_CBWR", "AUTO,STRICT")
