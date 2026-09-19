# natlang

A research project: fine-tune a very small language model (LFM2.5-350M) to be
the **interpreter** of a natural-language programming language. Programs,
state, and results live in one typed object tree. A folder-like `Lambda` node
holds instructions, typed inputs, and a typed result; the model reduces it by
reading, editing, copying, and triggering sub-lambdas, with exact work done by
sandboxed TypeScript.

## Documents

| File | Contents |
|------|----------|
| `PLAN.md` | Thesis, language design, agent interface, runtime, phases, open questions register |
| `TYPES.md` | Type system, validation, write-time typing, how validation feedback reaches the model |
| `TRAINING.md` | Use cases, skill taxonomy, datasets, teacher models, training recipe, evaluation |
| `SYNTHETIC_DATA.md` | Detailed designs and prior art for the fifteen synthesized datasets |
| `spec/` | The normative language specification (Phase 0) |
| `conformance/` | The conformance suite: small programs, one per language construct |
| `natlang/` | The harness (Python) |

Status: Phase 0 (specification, conformance suite) done. Phase 1 (harness) in
progress: the typed tree, validator, actions, sandbox, and combinators work
and are tested without a model; model-backed decoding is next.

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[js,dev]'
.venv/bin/python -m pytest -q          # harness scripts + canonical trace replays
python3 tools/check_conformance.py    # static checks on the suite
```

Running a program with a model (needs Docker with the NVIDIA runtime):

```
mkdir -p models && curl -L -o models/LFM2.5-350M-Q8_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf
scripts/serve.sh &                      # llama.cpp CUDA server on 127.0.0.1:8080; stop: docker stop natlang-llama
.venv/bin/python scripts/baseline.py 01 02 06      # tool surface + native constrained decoding (defaults)
.venv/bin/python scripts/baseline.py --decode server 01   # the server's own tool calling, unconstrained
.venv/bin/python -m natlang run conformance/programs/06-map-with-rubric.yaml --trace trace.jsonl
```

Teacher (Ternary Bonsai 2 27B, needs ~6 GB of free host RAM in addition to the GPU):

```
docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker
# binaries: PrismML-Eng/llama.cpp release, CUDA 12.8 tarball, unpacked into vendor/prism/bin
# model:    prism-ml/Ternary-Bonsai-2-27B-gguf  Ternary-Bonsai-2-27B-PTQ1_0.gguf  -> models/
# template: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit chat_template.jinja -> models/templates/Ternary-Bonsai-2-27B.jinja
scripts/serve_bonsai.sh 8081 &          # stop: docker stop natlang-bonsai
.venv/bin/python scripts/baseline.py --decode server --server http://127.0.0.1:8081 --thinking 256 --temperature 0.7 --verbose
.venv/bin/python scripts/paraphrase.py --server http://127.0.0.1:8081
.venv/bin/python scripts/generate.py --n 1000 --out data/ref-v0.jsonl
```
