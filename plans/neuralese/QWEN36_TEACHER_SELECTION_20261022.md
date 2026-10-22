# Qwen3.6 35B teacher selection — 2026-10-22

The owner selected Qwen3.6 35B as the continuing self-hosted teacher, superseding
Flint and the earlier Qwen3.8 selection. The exact upstream model is
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

Use generation sandbox `78530f71-08fc-4f57-ace9-7fbad64aeeb8`; no extra allocation.
Stage the [NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
by its actual immutable Hub revision, verify every artifact, and use the existing
Blackwell serving runtime where compatible. The
[official vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-35B-A3B.yaml)
provides RTX PRO 6000 settings. Runtime package version alone does not prove
working generation: verify model loading and actual task execution.

Preparation started without interrupting the existing Flint actor or either
learner. Actual preparation owner, download bindings and serving results live in
`/work/generation/qwen36-35b-teacher-20261022/` on the generation sandbox;
DGX launch receipt is
`/home/werg/data/dgxops/qwen36-teacher-preparation-launch-root-20261022.json`.
This document records selection and preparation, not a completed cutover.

At cutover, bind the new teacher descriptor and execution identity to a new
guardian epoch, preserve in-flight predecessor trajectories, and update export,
admission and supervision to include the new source. Keep the writer-occurrence
runtime correction. Admit verified complete positive trajectories through the
existing source, protected-split, classification and transport checks. Retain
usable earlier data under its original teacher identity. Keep both learners on
their evolving complete recipes with their optimizer, RNG and exposure intact.
