# Build against the installed PyTorch/CUDA ABI; retain the baseline image for comparison.
ARG BASE=natlang-train
FROM ${BASE}
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}
RUN pip install --no-cache-dir --no-deps --no-build-isolation causal-conv1d==1.7.0
