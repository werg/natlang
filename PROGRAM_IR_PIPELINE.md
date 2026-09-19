# Program IR and reproducible training builds

The durable data is **semantic program IR** (`natlang.program/1`), not a tool
trace or a model-formatted SFT pair. Each JSONL row has a stable program ID,
source and split provenance, license, source IDs, and a `semantics` object.
It contains source evidence, instructions or utterances, typed gold states or
answers, and operation dependencies. It contains no tool schema, system prompt,
assistant turn, chat template, or serialized model tokens.
Decision adapters retain only `question_key` from source metadata; teacher
model names, probability vectors, and raw Jev provider responses stay in the
source files and do not enter IR.

All sources pass through `scripts/build_program_ir.py` and
`scripts/materialize_ir.py`. The latter lowers an IR record with
`scripts/program_ir.py`, runs the reference policy through the real natlang
runtime, verifies the final value, and emits structured `messages`, `tools`,
and `target` turns. The trace manifest records the IR hash and harness source
hashes. These traces can be discarded and rebuilt when prompts, tools, policy,
or runtime change. Only after that should `scripts/export_sft.py` render a
specific model's chat template. Its output is model specific and disposable.
`export_sft.py` now records a renderer manifest and rejects resume with a
different template identity. Its `--end-token` and `--template-id` options let
the same structured turns be rendered with another compatible llama.cpp
template. It checks that the rendered tool call extends the rendered prompt,
so an incompatible template fails instead of silently making bad pairs.

The existing `data/external_pilot/*-traces.jsonl` and `*-sft.jsonl` made
before this change are **not canonical training sources**. The old traces
encode the harness at generation time. The old SFT pairs additionally encode
the then current llama.cpp chat template. Synthetic reference shards
from `scripts/generate.py` have a separate partial adapter. See
[SYNTHETIC_IR_MIGRATION.md](SYNTHETIC_IR_MIGRATION.md) for the exact replay
limit and the remaining semantic work. An old trace must not be treated as a
portable source program merely because its final answer verifies.

## Build commands

```bash
.venv/bin/python scripts/build_program_ir.py decisions \
  data/external_pilot/nanojev-stage1-curated-tasks.jsonl \
  data/external_pilot/typed-tasks.jsonl \
  data/external_pilot/sales-scale-labeled.jsonl \
  --out data/external_pilot/curated-v2.ir.jsonl

.venv/bin/python scripts/build_program_ir.py scone \
  --archive data/external_pilot/scone.zip \
  --out data/external_pilot/scone-v1.ir.jsonl

.venv/bin/python scripts/build_program_ir.py sgd \
  --dataset data/external_pilot/sgd \
  --out data/external_pilot/sgd-v2.ir.jsonl

.venv/bin/python scripts/build_program_ir.py finqa \
  --dataset data/external_pilot/finqa \
  --out data/external_pilot/finqa-v1.ir.jsonl

.venv/bin/python scripts/build_program_ir.py clevr \
  --archive data/external_pilot/CLEVR_v1.0_no_images.zip \
  --every 100 --out data/external_pilot/clevr-v1.ir.jsonl

.venv/bin/python scripts/materialize_ir.py \
  data/external_pilot/curated-v2.ir.jsonl \
  data/external_pilot/curated-v2-traces.jsonl.gz

# Large builds use atomic gzip shards and resume completed parts:
.venv/bin/python scripts/materialize_ir_shards.py \
  data/external_pilot/sgd-v2.ir.jsonl \
  data/external_pilot/sgd-v2-shards --workers 8 --shard-size 100

# Rebuild all synthetic families in the current architectural mix.
# The manifest records historical_match=false when the original sources drifted.
.venv/bin/python scripts/build_synthetic_ir.py \
  --manifest data/ref-v8-arch-seed73/manifest.json \
  --out data/external_pilot/synthetic-all-current.ir.jsonl \
  --allow-source-drift --workers 8

# Freeze executable failure and support cases with outcome contracts.
.venv/bin/python scripts/build_scenario_ir.py failures --seed 81 --groups 100 \
  --out data/external_pilot/failures-contract.ir.jsonl
.venv/bin/python scripts/build_scenario_ir.py support --seed 111 --groups 1000 \
  --out data/external_pilot/support-s111-complete.ir.jsonl

# Keep proposal reviews as linked semantic records, rendered from fresh base turns.
.venv/bin/python scripts/build_scenario_ir.py support --seed 113 --groups 1000 \
  --out data/external_pilot/support-s113-reviewbase.ir.jsonl
.venv/bin/python scripts/build_review_ir.py \
  data/agent-honesty-reviews-s113.jsonl \
  data/external_pilot/support-s113-reviewbase.ir.jsonl \
  data/external_pilot/reviews-s113-linked.ir.jsonl
.venv/bin/python scripts/materialize_review_ir.py \
  data/external_pilot/reviews-s113-linked.ir.jsonl \
  data/external_pilot/support-s113-reviewbase.ir.jsonl \
  data/external_pilot/reviews-s113-linked-traces.jsonl.gz
```

The IR adapters retain original source splits. `materialize_ir.py` verifies
each program against its gold answer and stops on a mismatch. Train, dev, and
test must remain separate outputs. For SCONE, use the full train TSV files,
which have each intermediate state; `train-orig` masks some intermediates.
For SGD, each service within a dialogue is a state sequence with a structured
`{active_intent, requested_slots, slot_values}` state and the service schema
in the task context. For FinQA, only numeric programs whose operations parse
and reproduce `exe_ans` are retained; table aggregation operations remain
outside this first adapter.
For CLEVR, the no-image archive supplies scene graphs and functional
programs. The adapter checks each program against the published answer, then
stores the natural-language question, compact scene graph, and program nodes.
The `--every 100` sample spans the full question order and yields 7,000 train
programs; the analogous validation sample has 1,500 programs.
The current harness allows at most 16 locals. CLEVR graphs with at most 16
nodes are emitted as one `run_code` and write per node; larger graphs are
compiled into one exact code expression. The IR keeps the full node graph in
both cases, so a later harness can choose a different execution plan.

## Migration status

The prior curated NanoJev, Typed Decisions, and labeled sales sources have
been rebuilt into one IR file: 4,611 programs and 25,328 verified turns. The
full accepted sales labels have also been rebuilt: 22,419 programs and
141,052 verified turns. No further Jev calls were needed. SCONE train has
11,198 state sequences and 179,190 verified turns. FinQA train has 6,036
supported numeric programs and 24,722 verified turns. SGD train has 29,642
service sequences in IR, with 1,000 programs and 27,147 turns verified in a
resumable sharded build. CLEVR has 7,000 sampled train programs and 130,552
verified turns in 70 compressed shards. Dev or
validation IR files for all four new sources are separate from train. The original source files and IR
should be retained so new renderer and harness versions can regenerate
trajectories.
The Kaggle SQL injection pilot was also migrated as 90 programs and 360
verified turns in `sql-research-v1`; its source license is recorded as
`unknown`, so it remains outside the license-gated training mix.
The smaller curated sales file overlaps the full sales file. To form a mix,
use `direct-core-v2.ir.jsonl` for NanoJev and Typed Decisions, plus
`sales-all-v2.ir.jsonl`; do not concatenate `curated-v2.ir.jsonl` with full
sales. `scripts/audit_program_ir.py` rejects duplicate program IDs and groups
that cross splits. The audited set of train and held-out IR files has 85,791
programs and no duplicate IDs or cross-split groups. `direct-core-v2` also
materialized 1,939 programs and 9,783 turns. Across the nonoverlapping train
sources materialized so far, there are 49,592 programs and 512,446 verified
turns; SGD's other 28,642 train programs remain in IR for later builds.

The current architectural synthetic generator has 10,000 programs and
280,472 verified turns in IR-backed shards. Of those, 453 programs contain
provisional generated-text gold, which SFT export excludes by default. Failure
and support cases add 17,200 `lambda_scenario` programs with explicit outcome,
effect, and action contracts. The combined execution IR audit found 27,200
unique program IDs and 11,100 isolated source groups. A separate seed 113
support base and 71,000 linked proposal-review records also pass the combined
uniqueness and split audit. The review records keep their contrast labels and
are rendered from newly executed base histories. SFT export uses the shared
scenario source group as `program_id` for execution and review turns, so a
subsequent program-level split keeps those contrasts together.

Changing a harness or prompt creates a new trace build with a new manifest.
Changing a chat template creates a new SFT build from those traces. Never
append records made under different harness or template versions to one file.
The shard builder refuses to resume if the IR, harness hashes, recovery rate,
or shard size changed. It publishes each compressed part only after every
program in that part passes verification.

Source references: [SCONE](https://nlp.stanford.edu/projects/scone/),
[Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue),
[FinQA](https://github.com/czyssrs/FinQA), and
[CLEVR](https://web.eecs.umich.edu/~justincj/clevr/).
