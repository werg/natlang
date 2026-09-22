# Teacher execution setup

> **Current path:** use `scripts/run_teacher_generation.sh`. It collects and
> materializes with the native Node/TypeScript runtime and resumes atomic jobs.
> The Python commands below document historical probes and completed backfills;
> do not use them for new teacher data.

The teacher is Ternary Bonsai 2 27B PTQ1_0, served by llama-server on port
8081. The probes record the actual `/v1/models` response, program, prompt,
tool transcript, runtime outcome, effects, and audit results. Teacher probes
are quarantined under `runs/`; they never append training references.
`scripts/serve_bonsai.sh` defaults to a 32,768-token server context; the
previous 12,288-token launch cap was an operational choice, not a limit of the
model weights. Every long-run trace records request timing and, for new runs,
prompt/completion token usage and the offered tool-schema size. Inspect these
before increasing context again: a large context should support long work, not
hide repetitive prompts or oversized tool menus.

The native coverage driver defaults to one worker, cache-stable tool schemas,
and 24-turn / 48-message continuation segments. One Bonsai decode saturates the
GPU, while two long contexts exceeded the server's 5 GiB host-memory cgroup in
earlier runs. The runtime still checks the exact current schemas; only the
model-facing path, line-number, and function-name hints are stabilized so llama-server
can reuse their common prompt prefix. The larger segment is deliberate: six-turn
segments cut ordinary nested functions and caused the teacher to reconstruct an
already-started repeat loop after continuation. The 12-turn setting completed the
same dependency-planner case correctly in 5 minutes 19 seconds with 82% prompt-cache
reuse; the previous run remained unfinished after 46 minutes. Continuations remain
enabled for genuinely long work, and checkpoint notes identify an unfinished loop's
accumulator and completed rounds.

## Keep the execution path simple

- Use `natlang/prompts/tools_explicit.md`, temperature 0, low reasoning
  effort, and a 256-token thinking budget.
- Teacher generation uses persistent typed scope through `eval`. Compute and
  update values directly in TypeScript, call helpers with ordinary awaited
  positional arguments, and use `mark_lines` after completing each instruction.
  The final eval expression that matches the declared return type completes the
  function. Use blocker and error actions to report missing information or invalid work.
- Keep compact state and no self-review. The established leaf collector uses
  caller validation feedback. The application evaluation harness uses local
  validation feedback so rejected writes remain visible to the model for
  repair; rejected and corrected turns are retained in raw audits.
- A normal assistant reply signals successful completion. The harness checks
  the final result and completed line marks at that boundary.
- Missing optional inputs are explicitly identified both in the opening state
  and during scope inspection. An actual empty string remains a valid supplied value.

These are interface clarifications and validation rules, not a deterministic
interpreter for the program's instructions. The model still chooses its actions.

## What the teacher tests exposed

| Run | Result | Finding |
| --- | --- | --- |
| `teacher-behavior-s881.json` | stopped after two cases | Teacher changed explicitly required Text into Num. |
| `teacher-behavior-prompt-s881.json` | six exploratory cases | A longer admonition did not fix wrong bindings or malformed values. |
| `teacher-behavior-compact-s881-rescored.json` | 14/16 | Missing write payload and wrong formal parameter name. |
| `teacher-behavior-final-s882-audit-v3.json` | 14/16, one wrong success | Missing input treated as empty; missing write payload. |
| `teacher-simple-focused-s883.json` | 5/6 | Explicit JSON-text transport and `done` worked; map duplicated the item input. |
| `teacher-simple-full-s884.json` | 18/21, one wrong success | Missing direct read still returned an ambiguous placeholder; map binding and quoted fold initializer failed. |
| `teacher-simple-focused-s885.json` | 4/4 | Missing versus empty, map, and fold pass after the targeted fixes. |
| `teacher-simple-full-s886.json` | 19/21, one wrong success | Teacher redirected an impossible direct return into a record field; another case attempted a premature error with truncated tool arguments. |
| `teacher-destination-effects-s886.json` | 8/8, zero wrong successes | Revised prompt passes both affected failure cases and their valid controls, on the original group and one fresh group. |

Schema isolation probes reproduced unexpected `{ "value": ... }` objects with
an untyped tool parameter. Explicit JSON shapes improved scalar handling but
did not reliably handle heterogeneous payloads. An explicit string transport
for JSON values fixed the tested record/list cases without a compatibility
unwrapper. This is evidence about the teacher/server interface; it does not
identify the model, parser, or constrained generation as the sole cause.

The audit was also corrected: a faithful attempted operation that encounters a
genuine type conflict is an acceptable failure. The model need not anticipate
every conflict by calling `report_error`. Redirecting the call, supplying a
different value, or reporting success still fails. Equivalent numeric fold
initializers are accepted whether passed by reference or literal.

After s886, the prompt and tool description explicitly state that an exact
destination cannot gain a field suffix to make types fit. The prompt also
clarifies ordered execution before a later conflict and requests a short error
explanation. The earlier field-copy example was removed from the general write
description. These changes address observed failures; their effectiveness must
be measured rather than assumed. The full s886 run used the preceding prompt,
preserved verbatim in its artifact.

The targeted follow-up passed all eight cases. Impossible direct bindings
attempted the requested destination and failed validation; valid bindings
succeeded. Required effects occurred before the later reported type conflict,
and the matching success cases completed. This addresses the observed failures
on those instances; the latest wording has not had another complete 21-case
run or a broad unseen-template evaluation. Keep per-sample admission checks.

## Reproduce and inspect

```bash
.venv/bin/python scripts/teacher_behavior_probe.py --seed 886 --temperature 0 --reasoning-effort low --json-text-values --system-file natlang/prompts/tools_teacher_compact.md --out runs/teacher-new-check.json
.venv/bin/python scripts/teacher_leaves.py --families cb_shopkeeper cb_webserver --n 4 --seed 886 --limit 4 --dry-run --audit-out runs/teacher-leaves-new-check.jsonl
```

Use fresh output paths. `teacher_leaves.py` uses this setup by default, records
every accepted/rejected attempt, and only admits completed outputs that pass
the existing leaf checks. `--dry-run` performs the same checks without admission.
Those checks combine crisp constraints and model judgments; they are not proof
of semantic correctness. A passing fixture suite likewise does not establish
that all teacher-generated programs are trustworthy.

The student behavior corpus is documented in [AGENT_SUPPORT.md](AGENT_SUPPORT.md).

The real collector dry run `runs/teacher-leaves-simple-s886.jsonl` passed all
four selected shopkeeper leaves with the revised prompt, including its crisp
checks and teacher judgments; zero references were admitted. This verifies
that collection path for those examples, not all generative leaf families.

## Reference backlog and first real backfill

The current-code 10,000-program synthetic IR snapshot has 453 provisional
programs with generated-text template stand-ins. Across them are 404 distinct
missing leaf keys: 264 `say`, 139 `page_content`, and one `submission_page`.
The reference bank initially had 33 entries, none matching these frozen keys.
A webserver dry run (`runs/teacher-webserver-dry-s886.jsonl`) passed one page
fragment and admitted none.

The first seed-73 backfill used the `v7_arch` mix and the first 300 programs,
whose 16 missing keys matched the frozen IR exactly in a replay check. Its
audit is `runs/teacher-leaves-ir-backfill-s73-first8.jsonl`: eight of eight
attempts passed the collector's checks and were admitted. The bank now has
41 entries; eight match the frozen IR. If the same program inputs were rebuilt
using these references, 11 of 453 provisional programs would become eligible,
leaving 442 programs and 396 distinct missing keys. The frozen IR and traces
still carry their original template outputs and provisional flags. A new IR
build and materialization, with new manifests, is required to incorporate
new references. The old historical v8 shard is not an exact replay because
its original generator snapshot is unavailable.

Before a full teacher pass, compare the generated task keys against the frozen
IR key inventory. Source drift can make a seed replay produce different keys.
Preserve rejected attempts and judge results in the audit, and sample accepted
outputs by function. The eight-case pilot establishes the collector path on
those inputs, not the acceptance rate or quality of all remaining leaves.

The collector can now read the exact missing cases from the frozen IR, without
replaying generator seeds:

```bash
.venv/bin/python scripts/teacher_leaves.py \
  --ir data/external_pilot/synthetic-all-current.ir.jsonl \
  --limit 16 --audit-out runs/teacher-leaves-ir-batch.jsonl
```

The current bank leaves 396 distinct keys in that snapshot (259 `say`, 136
`page_content`, one `submission_page`). Repeated runs skip admitted keys; use a
fresh audit path for each run. A rebuilt IR is still required to replace frozen
template gold with accepted references.

### Refreshing the frozen seed-73 corpus

The seed-73 source programs are frozen. Regenerating them from the current
generator can change their inputs, so apply checked references to that IR
directly. The refresh replays each affected program to update its expected
value and ordered effects, then verifies the result through the runtime. It
writes a new IR and hash manifest; it never edits the source IR.

```bash
.venv/bin/python scripts/audit_provisional_leaves.py \
  --ir data/external_pilot/synthetic-all-current.ir.jsonl \
  --out data/external_pilot/provisional-leaf-audit.json
.venv/bin/python scripts/refresh_synthetic_references.py \
  data/external_pilot/synthetic-all-current.ir.jsonl \
  data/external_pilot/synthetic-all-refreshed.ir.jsonl
.venv/bin/python scripts/audit_program_ir.py \
  data/external_pilot/synthetic-all-refreshed.ir.jsonl
.venv/bin/python scripts/materialize_ir_shards.py \
  data/external_pilot/synthetic-all-refreshed.ir.jsonl \
  data/external_pilot/synthetic-all-refreshed-shards \
  --workers 4 --shard-size 100
```

Use a new IR and shard destination after the reference bank changes. The SFT
exporter excludes `provisional_gold` turns by default. To finish the backfill,
run `teacher_leaves.py --ir` against the original frozen IR with a fresh audit
path for each pass, then repeat the refresh and materialization. The teacher
collector skips keys already admitted to `data/leaf_references.jsonl`. Once the
audit reports zero missing keys, add `--require-complete` to the refresh command
so an incomplete corpus cannot be mistaken for the final version.

For a long teacher pass, `scripts/finalize_synthetic_backfill.py` can wait for
its process, require the expected number of audit rows, then run the audit,
refresh, IR audit, and sharded materialization automatically. Pass the teacher
PID, its `/proc/PID/stat` start tick (field 22), the pass's original missing-key
count as `--expected-attempts`, and its unique `--teacher-audit` path. The
resulting filenames include the reference-bank hash. Its summary records
`complete: false` and the remaining provisional count if some leaf attempts
were rejected. A failed or truncated teacher pass publishes no snapshot.

The first long frozen-IR pass stopped after 14 of 394 attempts. Its watcher
correctly refused to publish (`runs/finalize-synthetic-backfill-pass1.log`).
The resumed pass used `--turn-tokens 1600`, started with 387 missing keys,
and wrote `runs/teacher-leaves-frozen-s73-extended-20260919.jsonl`. The
original 700-token cap caused six responses to stop at `finish_reason=length`;
four responses in the resumed audit also hit the 1600-token cap. The watcher
(`runs/finalize-synthetic-backfill-uncapped.log`) then ran another pass without
a separate per-turn cap, writing
`runs/teacher-leaves-frozen-s73-uncapped-retry-20260919.jsonl`. That snapshot
remained incomplete; the follow-up review below closed its remaining keys.
All pass audits remain available; admitted references are shared through
`data/leaf_references.jsonl`.

### Frozen-leaf rejection review, 20 September 2026

The 387-attempt resumed pass admitted 299 references and rejected 88. An
uncapped retry recovered 17 of the 18 unfinished cases; the last HTML page
needed 4,125 generated tokens across two turns and completed after removal of
the 4,000-token episode budget. The other 70 cases had completed `say` writes.
Many were rejected by the Bonsai yes/no semantic judge, whose short question
conflated a customer's rejected offer with the shopkeeper's counter-offer and
sometimes rejected valid sale dialogue. A revised question also produced false
positives, so its answer is advisory for dialogue. The follow-up audits and
manual decisions are in `runs/teacher-leaves-*20260920*.jsonl`.

All 70 `say` keys were filled after manual review and targeted retries. The
final reference bank has 437 distinct keys, with no duplicates, and the
original frozen IR has zero missing generative-leaf keys. `teacher_leaves.py`
now records `say` results for review by default; `--admit-say` is an explicit
override. Runtime inference has no default episode token, turn-count, or
wall-clock cutoff. The teacher collector likewise leaves each leaf's wall-clock
time and response length open unless `--max-seconds` or `--turn-tokens` is
supplied.

The complete reference bank has SHA-256 prefix `702a03e8`. Applying it to the
frozen seed-73 IR with `--require-complete` produced
`data/external_pilot/synthetic-all-complete-702a03e8.ir.jsonl` and its
manifest: 10,000 eligible programs, 453 refreshed programs, 929 replaced leaf
cases, and zero provisional programs. `audit_program_ir.py` passed. The
verified training traces are in
`data/external_pilot/synthetic-all-complete-702a03e8-shards`: 100 compressed
shards, 280,472 eligible turns, and 100,647 episodes. All shards passed gzip
integrity checks. Eight materialization workers caused a two-second effectful
JavaScript timeout under contention; the successful run used four workers.

Input variation from that frozen corpus is built separately with
`scripts/augment_map_inputs.py`. Seed 1 produced 376 map/map-count variants
from already labeled tickets; it changes item order and list length, then
recomputes the exact expected value from each program's frozen leaf oracle.
All 376 passed reference-runtime replay, yielding 4,151 eligible turns. Each
variant retains its parent's training `program_id`, so a program split keeps
the original and its input variant together. The variant IR and verified
shards are `data/external_pilot/synthetic-map-input-variants-s1.ir.jsonl` and
`data/external_pilot/synthetic-map-input-variants-s1-shards`.

The phase-one LFM-rendered bundle is `data/synthetic-phase1-s73.sft.jsonl`:
280,472 original turns plus 4,151 input-variant turns, or 284,623 pairs in
10,000 program groups. The original and variant template hashes match, and
every variant group is present among the originals. A separate seed-74
`v7_arch` run from a pinned checkout added 10,000 programs and 315,850
reference turns in `data/external_pilot/synthetic-s74-reference-shards`.
Its frozen IR is `data/external_pilot/synthetic-s74-frozen.ir.jsonl`;
`audit_program_ir.py` passed all 10,000. A provisional-leaf audit found 188
distinct new teacher keys across 195 programs. The first teacher pass is
recording attempts at `runs/teacher-leaves-s74-pass1.jsonl`; those attempts
must be reviewed and admitted before refreshing this IR for a complete
synthetic SFT export.

## End-of-turn completion

## Teacher trajectories to the next student corpus

`scripts/prepare_teacher_training.py` selects exact teacher trajectory IR rows
whose output matches `data/leaf_references.jsonl`. It applies the later
rejudgments and manual decisions by original audit file and line, with a
manual rejection overriding an automated approval. It replays each selected
choice in the current harness before writing structured turns. The source IR
is immutable. Older audits that lack a leaf definition can recover it from the
frozen program IR through `--program-ir`; the manifest records that source.

The current reviewed leaf set is `data/teacher-leaf-training-v4.jsonl`:
402 trajectories and 868 turns. Its manifest lists 35 bank entries with no
matching captured teacher trajectory. `data/teacher-leaf-sft-v2.jsonl` is the
corresponding LFM2.5 SFT view, rendered by the live model template on port
8080. It retains the teacher's action order and includes captured reasoning
for 844 turns in the training completion. The other 24 turns had no recorded
reasoning. The source, selection decision, and trajectory digest are carried
through the SFT rows. Tokenization with the student tokenizer found 40 of 868
teacher pairs above the trainer's default 3,072-token limit (maximum 6,050).
Use at least `--max-len 6050` for a run meant to include every teacher turn;
the next run should use `--max-len 8192`, subject to its GPU memory check.

For an incoming frozen program batch, collect accepted whole-program teacher
IR with the Node collector, then pass its JSONL to
`prepare_teacher_training.py --whole-ir`. Only attempts with successful
semantic trace admission are selected; replay checks them again. The same
`export_sft.py` step renders those turns for the next training corpus. Use a
new destination for each batch and concatenate the resulting SFT JSONL files
with the base corpus before starting the next fresh training run.
The collector's `--start` and `--limit` select disjoint ranges of an already
frozen JSONL file, so completed ranges can run while other program files are
still being generated. Preserve each range's raw trajectory and trace files.

```bash
node ts-host/scripts/teacher-collector.mjs \
  data/new-batch.ir.jsonl runs/new-batch.teacher.jobs runs/new-batch.teacher.ir.jsonl \
  --model-id Ternary-Bonsai-2-27B-PTQ1_0 --root-seed 907 --start 0 --limit 100
.venv/bin/python scripts/prepare_teacher_training.py data/new-batch.teacher.turns.jsonl \
  --whole-ir runs/new-batch.teacher.ir.jsonl
.venv/bin/python scripts/export_sft.py data/new-batch.teacher.turns.jsonl data/new-batch.teacher.sft.jsonl \
  --server http://127.0.0.1:8080 --template-id LFM2.5-350M --workers 8
```

One real smoke run on `data/external_pilot/synthetic-simple-seed73-current.ir.jsonl`
completed as `73:judge:4` with Bonsai. It passed trace admission, replayed as
two turns, and exported both turns with captured reasoning to
`data/teacher-program-bridge-pilot.sft.jsonl`. The source attempt and trace
are `runs/teacher-program-bridge-pilot.ir.jsonl` and its adjacent trace file.
This verified the whole-program bridge on one frozen program; each incoming
batch still needs its own admission and replay check.

The remaining 15 programs in that frozen set also completed and passed
admission, including an eight-item map/count program. Their raw IR is
`runs/teacher-program-simple-s73-remaining.ir.jsonl`. Replaying both files
produced 53 turns in `data/teacher-program-simple-s73.turns.jsonl`; the LFM
template export has 53 SFT pairs, all with captured reasoning. None exceeds
3,072 student tokens. `scripts/combine_sft.py` checks source template
identities and duplicate IDs while writing a hashed bundle. The current
combined teacher SFT file is `data/teacher-reviewed-bundle.sft.jsonl`: 921
distinct pairs from the reviewed leaf set and these 16 whole programs, 897
with reasoning. Its manifest pins the two source SFT hashes. Add the incoming
program batch's admitted SFT pairs to a new combined file with
`combine_sft.py` before the next fresh training run.

To collect more teacher trajectories from the current frozen corpus,
`scripts/select_teacher_programs.py` selects a reproducible round-robin batch
across available families. The seed-909 selection contains 96 distinct
programs across 24 families and is recorded at
`data/external_pilot/teacher-selection-balanced-s909.ir.jsonl`, with its
source hash and chosen IDs in the adjacent manifest. Its IDs do not overlap
the 16 already admitted whole-program teacher pilots. Run the teacher
collector over disjoint ranges of this file and retain rejected attempts for
diagnosis, without adding them to training.

For phased training, export the synthetic shards and their input variants with
the same LFM template used by the teacher bundle. Train phase 1 on the
synthetic SFT in a new run directory, then use `--merge-only` if needed to
produce that run's `merged` model. Start phase 2 on the reviewed teacher SFT
with `--model runs/<phase-1>/merged` and a *different* output directory. This
loads the phase-1 weights and creates a new LoRA adapter for the teacher
phase; a same-directory resume cannot change corpus identity. Use
`--max-len 8192` for phase 2 if the GPU memory check permits it. Keep
program-level holdout groups fixed across phases when comparing losses or
behavior; an in-phase random holdout alone does not enforce that.

The data-migration Bonsai pilot uses
`codebases/data_migration/scenarios/two_exports.json` with
`scripts/run_data_migration.py` in preview mode. Early traces showed that
Bonsai invented output field names (`source_customer_id`, then `decision`),
which the type checker rejected. A later attempt treated two same-email source
customers as separate `new` identities because the instructions only described
the database snapshot. The codebase now names every output field and explains
within-batch identity matching; the host validates decisions independent of
their reply order. `runs/data-migration-bonsai-pilot4.json` completed the
two-export preview: both mappings were correct, one same-email pair became
`new` plus `merge`, and the plan contains two customers and three orders with
no review items. Each pilot has its own SQLite file and trace directory; no
import was applied.

The one-case test-explorer pilot first wrote a bare ID list instead of its
`Selection` record. After the shape was clarified, it selected the
`missing-parent` case and ran the nested dependency planner; the independent
graph oracle found no violation. Its assessment then used invented record
fields and failed typing. With `Assessment`'s three field names stated
explicitly, replaying the saved observation completed with empty `findings`,
`unknowns`, and `followups` in
`runs/test-explorer-assess-replay.json`. This checks the current interface on
one case; the full explorer has not yet completed a second end-to-end run.

The terminal `done` tool has been removed from the model-facing surface and
reference policy. `done=N` on `write` and `call` still closes a numbered line.
The harness now checks open marked lines when the assistant ends with a normal
reply, and it records that reply in the transcript. Blank lines, comments, and
function declarations are not markable instructions.

Line closure is currently enforced only after marking has begun. Making marks
mandatory for every codebase episode needs reference-plan updates: nine
generator/IR test families still produce valid outcomes with no marks. Do not
invent closing marks at the end of those trajectories; encode the actual
done/skipped semantics when migrating them.

The four-case dry run `runs/teacher-leaves-reply-only-s73-first4.jsonl` used
the revised teacher prompt and passed 4/4 collector checks (three `say`, one
`page_content`), admitting zero references. No terminal `done` call appeared.
The run preceded the transcript recording fix, so its audit does not include
the final prose turn. This is a small smoke test, not a full teacher behavior
regression or evidence about student quality.

Teacher prose remains in collection audits as a diagnostic note. Student
reference trajectories now target an empty final assistant message: the
end-of-turn token alone signals successful completion after the runtime checks.
The native grammar accepts this empty turn; SFT export renders it as only the
assistant end token. Error and blocker reports retain their diagnostic text.
