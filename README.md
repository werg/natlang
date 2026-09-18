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

Running a program with a model (needs a llama.cpp `llama-server` serving a
GGUF of the model on port 8080):

```
.venv/bin/python -m natlang run conformance/programs/06-map-with-rubric.yaml --trace trace.jsonl
```
