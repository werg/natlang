# Shared neuralese training image for Pop (Ada, sm_89), DGX Spark (GB10, sm_121, arm64) and CoreWeave sandboxes
# (RTX PRO 6000 Blackwell, sm_120, x86_64: scripts/coreweave/build_image.sh): NVIDIA's PyTorch build
# plus the current releases of our Python dependencies. NGC's own torch/triton stay pinned by its pip
# constraints. Code is mounted, not copied: run with -v /home/werg:/home/werg and the repo as the working directory.
#   docker build -t natlang-neuralese:26.09 training/neuralese/docker
# Record the resolved image digest and installed package versions in every launch; BASE_IMAGE can pin the native
# architecture digest while keeping one shared implementation on both machines.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}
COPY requirements.txt /tmp/natlang-requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/natlang-requirements.txt
ENV PYTHONUNBUFFERED=1
