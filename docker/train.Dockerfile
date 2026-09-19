# Training image for the local proof run and for memory-constrained fine-tuning. It reuses a local image that
# already has PyTorch with CUDA (any recent NGC PyTorch image works: set BASE), and adds the small Python packages.
ARG BASE=chesst-zero-train:latest
FROM ${BASE}
RUN pip install --no-cache-dir "transformers>=4.55" "peft>=0.13" "accelerate>=1.0" sentencepiece
WORKDIR /work
ENTRYPOINT []
