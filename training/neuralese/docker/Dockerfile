# Shared neuralese training image for Pop (Ada, sm_89) and DGX Spark (GB10, sm_121): NVIDIA's PyTorch build
# plus the current releases of our Python dependencies. NGC's own torch/triton stay pinned by its pip
# constraints. Code is mounted, not copied: run with -v /home/werg:/home/werg and the repo as the working directory.
#   docker build -t natlang-neuralese:26.09 training/neuralese/docker
# Record the resolved image digest and installed package versions in every launch; BASE_IMAGE can pin the native
# architecture digest while keeping one shared implementation on both machines.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir --upgrade transformers peft kernels accelerate safetensors pytest \
      tree-sitter==0.26.0 tree-sitter-typescript==0.23.2
ENV PYTHONUNBUFFERED=1
