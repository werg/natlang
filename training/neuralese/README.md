# natlang_neuralese: shared native training, serving and tuning

The shared PyTorch package behind the native Neuralese lines: shared positional KV and hybrid-state spans with
distinct producer/consumer markers (`model/shared_state.py`), the shared native SFT trainer and recipe loader
(`train/native_state_sft.py`, `train/recipe.py`), whole-graph RL with JOLT+ (`rl/`), serving (`serve/`), the
tuning system and learned optimizer (`tune/`), and quantized export (`export/`). The current line is LFM3
(`plans/neuralese/LFM3_RECIPE_20261028.md`); recipes live in `recipes/` and use `extends`/`overrides`.

| Path | Contents |
| --- | --- |
| `model/` | Backbone ports, native shared-state execution, native checkpoints and parameter precision |
| `data/` | Native graph schema, conversion and rendering, trajectory text and source readers |
| `train/` | Recipe loader, native SFT/preference trainers, optimizers (Muon/AdamW, stochastic rounding), QAT, EMA, DiLoCo |
| `rl/` | Whole-graph RL rounds, JOLT+ privileged self-teaching, score-centered objective, memetic programs |
| `serve/` | Native engine, HTTP server, guided generation |
| `tune/` | Program tunes, native soft states, LoRA mixtures/bases, learned optimizer |
| `eval/` | Task packets and cadence, native state/gradient/memory checks, throughput |
| `common/` | Shared hashing, JSON I/O, checkpoint storage, line backbones |

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
cd training/neuralese
../../.venv-neuralese/bin/python -m natlang_neuralese.train.recipe --help
```
