# Runtime for Prism ML's prebuilt llama.cpp fork binaries: CUDA 12.8 runtime + the two libraries they need.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
