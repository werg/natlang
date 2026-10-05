# natlang_neuralese: the S3 port reference

PyTorch reference of the Neuralese read and write ports on LFM2.5 ([S3 plan](../../plans/neuralese/S3_PORT.md)).

New training lineages start with the [shared declared recipe](../../plans/neuralese/TRAINING_RECIPE.md):
token identity and causal embedding distillation are prerequisites, followed by
separate runtime qualification before compression and recurrence.

| Path | Contents |
| --- | --- |
| `model/lfm2_port.py` | Layer-range execution of the HF LFM2 weights with snapshot-able caches; fast path (SDPA native GQA and causal handling, preallocated copy-on-write KV without autograd, hub `causal-conv1d` for prefills) and the original reference path |
| `model/heads.py` | Feedback projection, stop head, content projection as a payload distribution (mean, per-dimension log-sigma, temperature-gated sampling, log-likelihood, KL), interface norm |
| `write.py`, `read.py` | The single write procedure (`temperature`, default 0) and the read port |
| `data/` | Port-record loader (`natlang.port-record/1`), per-model rendering (chat and Natlang forms), span examples, fixtures |
| `train/` | Training-time execution (parallel scheduled sampling, full unroll), phase A–D losses and schedules, the resumable trainer, the smoke run |
| `eval/` | The harness (payload ablations, stopping, representation monitors, temperature sweep, cache agreement, latency) and the throughput benchmark |
| `laws.py` | Law-measurement hooks (objectives arrive in S5) |

## Environment

A dedicated venv (gitignored) at the repository root:

```sh
uv venv --python /usr/bin/python3.12 .venv-neuralese
uv pip install --python .venv-neuralese/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==2.11.0 transformers==5.18.0 peft==0.21.2 kernels==0.17.2 accelerate safetensors numpy pytest \
  tree-sitter==0.26.0 tree-sitter-typescript==0.23.2  # guided generation (serve/guidance.py)
```

On the GB10 (sm_121, aarch64) torch's flash and cuDNN SDPA backends work with `enable_gqa` and lower-right causal masks; the memory-efficient backend does not take GQA. The hub `kernels-community/causal-conv1d` kernel loads for sm_121 and is pinned by revision so it works offline. A separate flash-attention build is not needed.

## Commands

```sh
.venv-neuralese/bin/python -m pytest -q tests/neuralese
cd training/neuralese
../../.venv-neuralese/bin/python -m natlang_neuralese.train.smoke --out RUN_DIR --device cuda
../../.venv-neuralese/bin/python -m natlang_neuralese.eval.bench --prefix 2048 --block 32 --batch 1
```
