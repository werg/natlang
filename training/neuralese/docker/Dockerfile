# Neuralese training image for the DGX Spark (GB10, sm_121): NVIDIA's PyTorch build (native sm_121, tuned cuBLAS/
# cuDNN) plus the latest releases of our Python dependencies. NGC's own torch/triton stay pinned by its pip
# constraints. Code is mounted, not copied: run with -v /home/werg:/home/werg and the repo as the working directory.
#   docker build -t natlang-neuralese:26.09 training/neuralese/docker
FROM nvcr.io/nvidia/pytorch:26.09-py3
RUN pip install --no-cache-dir --upgrade transformers peft kernels accelerate safetensors pytest \
      tree-sitter==0.26.0 tree-sitter-typescript==0.23.2
ENV PYTHONUNBUFFERED=1
