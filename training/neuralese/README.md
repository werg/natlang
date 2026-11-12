# natlang_neuralese: shared native training, serving and tuning

The shared PyTorch package behind the native Neuralese lines: shared positional KV and hybrid-state spans with
distinct producer/consumer markers (`model/shared_state.py`), native checkpoints and their loader
(`train/checkpoint_policy.py`), the RL collector, scorer, task selection and held-out evaluation (`rl/`), serving
with the SGLang actor and the native gradient endpoints (`serve/`, `train/native_graph_loss.py`), the tuning system
(`tune/`), and quantized export (`export/`). Fine-tuning runs through the HF path
(`scripts/neuralese_data/exp0_sft.py`); the BF16 native SFT trainer and the w5 RL controller were removed on
2026-11-12 (git history keeps them).

| Path | Contents |
| --- | --- |
| `model/` | Backbone ports, native shared-state execution, native checkpoints and parameter precision |
| `data/` | Native graph schema, conversion and rendering, trajectory text and source readers |
| `train/` | Checkpoint loading/offload, native graph loss, optimizers, QAT, own-output supervision |
| `rl/` | Task rollouts and scoring, task selection, held-out evaluation, whole-graph replay objective |
| `serve/` | Native engine, HTTP server, guided generation |
| `tune/` | Program tunes, native soft states, LoRA mixtures/bases, learned optimizer |
| `eval/` | Task packets, ablation and FFN dispatch probes |
| `common/` | Shared hashing, JSON I/O, paths, line backbones |

## Environment

A dedicated venv (gitignored) at the repository root:

```sh
uv venv --python /usr/bin/python3.12 .venv-neuralese
uv pip install --python .venv-neuralese/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==2.11.0 transformers==5.18.0 peft==0.21.2 accelerate safetensors numpy pytest \
  tree-sitter==0.26.0 tree-sitter-typescript==0.23.2  # guided generation (serve/guidance.py)
```

On the GB10 (sm_121, aarch64) torch's flash and cuDNN SDPA backends work with `enable_gqa` and lower-right causal masks; the memory-efficient backend does not take GQA. The former Hub causal-convolution revision is invalid, so GB10 currently uses the reference `torch.nn.functional.conv1d` path. A Pop-only Dao-AILab source build for Torch 2.14/CUDA 13.4/SM 8.9 is tracked by artifact ID `causal-conv1d-pop-torch214-cu134-sm89-v1`; it remains opt-in until same-input CUDA fidelity and throughput are qualified. The H100 has a separate SM90 wheel for the same Torch/CUDA build (`causal-conv1d-coreweave-torch214-cu134-sm90-v1`), with exact measured operator output/input-gradient/weight-gradient parity and long-context speedups; activation stays explicit at a useful full-model handoff. Build-time provenance and later operator qualification are separate records. A separate flash-attention build is not needed.

## Commands

```sh
.venv-neuralese/bin/python -m pytest -q tests/neuralese
```
