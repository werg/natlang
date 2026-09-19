# natlang

A research project: fine-tune a very small language model (LFM2.5-350M) to be
the **interpreter** of a natural-language programming language.

A natlang program is a **pseudocode algorithm**: functions with typed
signatures, subroutine calls, `for each`, `repeat until`, `if`/`else`, local
variables, organised as a **code base** of `.nl` files (frontmatter plus a
pseudocode body; `somefun.nl` with an optional companion folder `somefun/`;
`.ts` files for exact functions; reuse through `uses` links). The author
states the structure. The model carries it out one small step at a time with
seven tools (`read`, `write`, `edit`, `run_code`, `call`, `mark_done`, `report_blocker`);
every function instance is a fresh short episode over a typed object tree.
Prompt-like tasks (judge, classify, extract, rewrite) are the leaves. The
harness provides memory, typing, a sandbox and I/O; it parses no instructions
and owns no control flow.

```
# examples/triage/main.nl (body)
function triage(tickets, rubric) -> Report
  labels   = for each t in tickets: classify(t, rubric)
  not_spam = for each l in labels: l is not "spam"            # exact: use code
  real     = select_by_flags(tickets, not_spam)
  flags    = for each t in real: is_urgent(t)
  ...
```

## Documents

| File | Contents |
|------|----------|
| `PLAN.md` | Thesis, language design, tool surface, runtime, data generation, phases, status and findings |
| `spec/SPEC.md` | The normative language specification (v0.2-draft); `spec/CODEBASES.md` gives the rationale for code bases and `call` |
| `TYPES.md` | Type system, validation, write-time typing, how validation feedback reaches the model |
| `TRAINING.md` | Use cases, skill taxonomy, datasets, the teacher's roles, training recipe, evaluation |
| `SYNTHETIC_DATA.md` | Detailed designs and prior art for the fifteen synthesized datasets |
| `examples/`, `codebases/` | Code bases on disk: `triage`; `nlprolog`, `highlighter`, `webserver`, `moderation`, `shopkeeper`, `legal_move`; the linkable crisp library `std/` |
| `conformance/` | The conformance suite and the harness scripts |
| `natlang/` | The harness (Python): tree, types, code bases, tool surface, constrained decoding, generators |

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[js,dev]'
.venv/bin/python -m pytest -q          # harness, code bases, surface, grammar, generator
python3 tools/check_conformance.py    # static checks on the suite and the spec's type expressions
```

Generating training data (CPU only; every turn is checked by the harness, by
the grammar of its own turn, and against the expected value):

```
.venv/bin/python scripts/generate.py --n 2000 --seed 1 --out data/ref.jsonl
```

The generator mixes leaf tasks, synthesized pseudocode programs (`natlang/gen/synth.py`) and the hand-written
code bases (`natlang/gen/codebases.py`); choose with `--families`.

The whole loop on one 8 GB GPU (the training image reuses any local PyTorch image: see `docker/train.Dockerfile`):

```
scripts/serve.sh &                                                          # the base model, for its chat template
.venv/bin/python scripts/export_sft.py data/ref.jsonl data/sft.jsonl        # pairs rendered exactly as at inference
docker build -t natlang-train -f docker/train.Dockerfile docker
docker run --rm --gpus all -v "$PWD:/work" -e HF_HOME=/work/models/hf natlang-train \
    python scripts/train_lora.py data/sft.jsonl runs/lora --steps 300         # LoRA, bf16, checkpointing: about 3 GB
# stoppable and resumable: `docker stop -t 120 natlang-train` writes a checkpoint; the same command continues;
# `--merge-only` exports runs/lora/merged from the latest checkpoint; `--fresh` starts over
scripts/to_gguf.sh runs/lora/merged models/natlang-350M-Q8_0.gguf
docker stop natlang-llama; scripts/serve.sh natlang-350M-Q8_0.gguf &
.venv/bin/python scripts/eval_turns.py data/ref.jsonl                       # next-turn accuracy per kind of turn
.venv/bin/python scripts/baseline.py                                        # whole programs, graded by their checks
```

A web server whose every request is interpreted by the model (`codebases/webserver`):

```
scripts/serve_web.py --port 8000 --server http://127.0.0.1:8080
```

Running programs with a model (needs Docker with the NVIDIA runtime):

```
mkdir -p models && curl -L -o models/LFM2.5-350M-Q8_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf
scripts/serve.sh &                      # llama.cpp CUDA server on 127.0.0.1:8080, official chat template; stop: docker stop natlang-llama
.venv/bin/python scripts/baseline.py 01 02 23      # tool surface + native constrained decoding (defaults)
.venv/bin/python -m natlang run examples/triage/main.nl --in tickets=./tickets/ --in rubric=./rubric.md
```

Teacher (Ternary Bonsai 2 27B on an 8 GB GPU; about 1 GB of host RAM):

```
docker build -t natlang-prism-runtime -f docker/prism.Dockerfile docker
# binaries: PrismML-Eng/llama.cpp release, CUDA 12.8 tarball, unpacked into vendor/prism/bin
# model:    prism-ml/Ternary-Bonsai-2-27B-gguf  Ternary-Bonsai-2-27B-PTQ1_0.gguf  -> models/
# template: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit chat_template.jinja -> models/templates/Ternary-Bonsai-2-27B.jinja
scripts/serve_bonsai.sh 8081 &          # stop: docker stop natlang-bonsai;  scripts/watch_bonsai.sh keeps it up
.venv/bin/python scripts/baseline.py --decode server --server http://127.0.0.1:8081 --thinking 512 --temperature 0.6 \
    --system-file natlang/prompts/tools_delegate.md --alias call=call_function --verbose 23
.venv/bin/python scripts/paraphrase.py --server http://127.0.0.1:8081     # paraphrases, kept only after a round trip
.venv/bin/python scripts/teacher_leaves.py --server http://127.0.0.1:8081  # references for leaves that generate text, kept only if checks pass
```
