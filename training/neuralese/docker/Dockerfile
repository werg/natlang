# Shared neuralese training image for Pop (Ada, sm_89), DGX Spark (GB10, sm_121, arm64) and CoreWeave sandboxes
# (RTX PRO 6000 Blackwell, sm_120, x86_64: scripts/coreweave/build_image.sh): NVIDIA's PyTorch build
# plus the current releases of our Python dependencies. NGC's own torch/triton stay pinned by its pip
# constraints. Code is mounted, not copied: run with -v /home/werg:/home/werg and the repo as the working directory.
#   docker build -t natlang-neuralese:26.09 training/neuralese/docker
# Record the resolved image digest and installed package versions in every launch; BASE_IMAGE can pin the native
# architecture digest while keeping one shared implementation on both machines.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}
# The shared text reader verifies zstd streams through the command-line decoder.
RUN apt-get update && apt-get install -y --no-install-recommends zstd \
 && rm -rf /var/lib/apt/lists/*
# Every GPU this image serves: Pop (8.9), CoreWeave RTX PRO 6000 (12.0), DGX GB10 (12.1). Read by JIT extension builds
# (torch.utils.cpp_extension); Triton kernels compile per device at run time. plans/COREWEAVE_HARDWARE_OPTIMIZATION.md.
ARG TORCH_CUDA_ARCH_LIST="8.9 12.0 12.1"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
COPY requirements.txt /tmp/natlang-requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/natlang-requirements.txt
# Fail the build when the base's PyTorch has no sm_120 code (the RTX PRO 6000; GB10's sm_121 runs it too, and Pop's
# sm_89 runs the sm_86 code NGC ships), and record the resolved versions inside the image (the pins of this build;
# launches record the image digest).
RUN python -c "import torch; arch = torch._C._cuda_getArchFlags().split(); assert 'sm_120' in arch, ('torch lacks sm_120', arch)" \
 && mkdir -p /opt/natlang && pip freeze > /opt/natlang/pip-freeze.txt
ENV PYTHONUNBUFFERED=1
