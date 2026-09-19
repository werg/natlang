# Synthetic corpus IR

The source of truth is the versioned `natlang.program/1` JSONL, not a tool
trace or model-formatted SFT pair. `scripts/build_synthetic_ir.py` regenerates
tasks from the current synthetic generator and freezes their source lambda or
codebase, initial inputs or fold events, typed gold, leaf oracle cases, source
operations, and expected effects. `scripts/program_ir.py` lowers these records
to today's reference policy. `scripts/materialize_ir.py` and
`scripts/materialize_ir_shards.py` run that policy in the real runtime and
verify the result. `scripts/export_sft.py` applies a chosen chat template only
after verification.

The synthetic adapter covers every family in the v8 architectural mix: simple
leaves, compositional reports, nested conditional functions and folds, and all
ten codebase families. The operation graph supports function calls, exact
calculations, typed assignments, conditional branches, repeats, local function
edits, blockers, and retries. It also records source-line relationships for
the synthesizer. Codebase cases embed the function definitions and, for open
folds, the initial state and event stream. Effectful cases state their
expected emissions; the order saga additionally records an acknowledgement
loss and expected unique deliveries.

The full current-code build from `data/ref-v8-arch-seed73/manifest.json` has
10,000 IR programs in `data/external_pilot/synthetic-all-current.ir.jsonl`.
Its manifest says `historical_match: false`: seed and family mix were reused,
but these are newly generated tasks. The old shard used an unavailable source
snapshot. The adapter refuses source drift unless `--allow-source-drift` is
given, and then labels the output accordingly. An exact conversion of old
traces is unnecessary for the current-generator corpus.

```bash
.venv/bin/python scripts/build_synthetic_ir.py \
  --manifest data/ref-v8-arch-seed73/manifest.json \
  --out data/external_pilot/synthetic-all-current.ir.jsonl \
  --allow-source-drift --workers 8

.venv/bin/python scripts/materialize_ir_shards.py \
  data/external_pilot/synthetic-all-current.ir.jsonl \
  data/external_pilot/synthetic-all-current-shards \
  --workers 8 --shard-size 100
```

Some `cb_shopkeeper` and `cb_webserver` leaves produce generated text when no
accepted teacher reference exists. The IR records the observed answer and
marks the entire program `contains_templates`. Materialized turns carry
`provisional_gold`; SFT export excludes those programs by default. In this
build, 453 of 10,000 programs are provisional. They can be rematerialized
after reference coverage improves.

The observed oracle tables for those two applications cover the calls made by
their generated case. A future policy that makes different leaf calls may need
additional reference answers. Other synthetic source oracles are captured
from the generator's latent world or from explicit rules. The operation graph
records the task's dependencies; the current choice of turns, tool calls,
line-marking style, system prompt, and chat template remains disposable.

The failure and agent-support execution cases now use `lambda_scenario` IR.
Their outcome contract records `done`, `error`, or `blocked`, the expected
value or explanation, ordered effects, required semantic actions, and exact
function destinations and input references where a tempting alternative would
change the task. `scripts/audit_trajectory_admission.py` checks candidate
actions against that contract; the materializer also executes the reference
in the runtime to verify value, outcome, and effects. An arbitrary teacher
trace still needs runtime replay before its claimed effects can be trusted.
The current case sets have 1,200 failure programs and 16,000 support programs.
The combined synthetic, failure, and support IR audit found 27,200 distinct
program IDs and no split-group conflicts.

Proposal reviews use a separate `proposal_review` IR kind because the target
is a judgment about a proposed action. Each record links to a base program
digest and turn, and stores the proposed action, verdict, reason, lesson IDs,
task feasibility, and shared contrast group. `scripts/build_review_ir.py`
migrated 71,000 frozen reviews from seed 113: 23,000 approvals, 41,000
withdrawals, 6,000 errors, and 1,000 blockers. Every contrast group contains
an approved and a challenged candidate. `scripts/materialize_review_ir.py`
re-executes the base program under the current harness, verifies each approved
proposal still matches its reference turn, and builds review messages with the
current review prompt. All 71,000 reviews were rendered to
`data/external_pilot/reviews-s113-linked-traces.jsonl.gz`. The full synthetic,
failure, support, review-base, and review-IR audit has 114,200 unique IDs.
Review labels remain constructed contrasts; confidence is deliberately unset.
