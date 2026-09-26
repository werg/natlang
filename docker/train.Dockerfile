# Training image for the local proof run and for memory-constrained fine-tuning. It reuses a local image that
# already has PyTorch with CUDA (any recent NGC PyTorch image works: set BASE), and adds the small Python packages.
ARG BASE=chesst-zero-train:latest
FROM ${BASE}
RUN pip install --no-cache-dir \
    transformers==5.5.0 peft==0.21.0 accelerate==1.15.0 bitsandbytes==0.50.2 \
    sentencepiece==0.2.2 unsloth==2026.9.7
# Linear-attention kernels for hybrid models such as Ling-3.0 (their modeling code imports fla); no extras, so the
# base image's CUDA build of torch and triton stays.
RUN pip install --no-cache-dir --no-deps fla-core==0.5.2 einops==0.8.2
WORKDIR /work
ENTRYPOINT []
