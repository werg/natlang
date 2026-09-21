# Training image for the local proof run and for memory-constrained fine-tuning. It reuses a local image that
# already has PyTorch with CUDA (any recent NGC PyTorch image works: set BASE), and adds the small Python packages.
ARG BASE=chesst-zero-train:latest
FROM ${BASE}
RUN pip install --no-cache-dir \
    transformers==5.5.0 peft==0.21.0 accelerate==1.15.0 bitsandbytes==0.50.2 \
    sentencepiece==0.2.2 unsloth==2026.9.7
WORKDIR /work
ENTRYPOINT []
