# Teacher trajectory IR

## Canonical native pipeline

New collection and materialization run entirely in Node on the canonical
TypeScript host:

```bash
node ts-host/scripts/teacher-collector.mjs \
  data/teacher/coverage-selection-s909.ir.jsonl runs/teacher.jobs \
  runs/teacher.native.jsonl --model-id MODEL --root-seed 909 \
  --limit 96 --workers 1 --segment-turns 24 --segment-messages 48
node ts-host/scripts/materialize-native-teacher.mjs \
  runs/teacher.native.jsonl data/teacher-native-turns.jsonl --replace
```

The collector stores the exact request, exposed reasoning, normalized model
choice, offered tools, action outcomes, trace identity, continuation boundaries,
and final exact admission verdict. The materializer accepts only admitted rows,
links every call to its ordered action ledger event, and emits both lossless
semantic decisions and standard `messages`/`tools`/`target` training fields.
It never replays a decision through the retired Python runtime. Checkpointed
segments reopen from durable typed state and a short note; earlier dialogue is
not copied into the next segment.

`scripts/run_teacher_generation.sh` performs this flow for every current
coverage program and Studio application, rescans for new sources between waves,
and resumes per-case atomic jobs after interruption. See
`ts-host/TEACHER_MATERIALIZATION.md` for the native record contract.

## Legacy trajectory migration

The teacher's decisions are durable source data. `scripts/teacher_leaves.py --ir`
records the complete server replies, exposed reasoning, pre-action message
context, offered tool schemas, proposed calls, self-reviews, attempted calls and tool results in its
audit. It writes a linked `natlang.teacher_trajectory/1` JSONL beside that
audit. The audit is retained even when a result is rejected or execution
raises an exception. The IR is independent of Bonsai's rendered chat tokens.

The IR links each leaf task by reference key to every matching program in the
frozen synthetic IR. Its ordered turns distinguish a **proposal** from an
**execution**: a rejected call remains visible, and a batched call that was
never executed is not made to look successful. The raw audit contains the
server's exact JSON response; the IR extracts content, any exposed reasoning,
canonical tool names, arguments, review verdicts, results, and outcome checks.
Each turn carries a digest of its raw response, and the IR records the audit
location and row digest. We retain the source spelling of each tool alongside
its canonical name, so a later migration can be revised without losing the
original choice.

Existing audits can be converted without rerunning the teacher:

```bash
.venv/bin/python scripts/teacher_trajectory_ir.py \
  runs/teacher-leaves-ir-backfill-s73-first8.jsonl \
  data/external_pilot/synthetic-all-current.ir.jsonl \
  runs/teacher-leaves-ir-backfill-s73-first8.trajectory-fresh.ir.jsonl
```

Behavior probes use the same trajectory IR without pretending to be leaf
references. Their known expected value, failure contract, required-call check,
and effects verdict stay attached to each trajectory. New probe runs write this
IR beside the JSON audit automatically; old probes can be converted:

```bash
.venv/bin/python scripts/teacher_probe_trajectory_ir.py \
  runs/teacher-behavior-reply-only-s886.json \
  runs/teacher-behavior-reply-only-s886.trajectory-fresh.ir.jsonl
```

Old audits contain the visible transcript and action log but did not capture
the server's reasoning field or every rejected turn's context. Conversion
marks those losses explicitly and keeps the complete ordered action log in
structured form, including calls absent from the old transcript. New collection
records both. If the server
does not expose reasoning, it cannot be recovered; the audit still preserves
every returned field and the model's visible explanations.
Some old reply-only audits also missed the final assistant turn. The IR marks
that gap, and the projection may synthesize an empty final turn only when the
recorded outcome was successful. Audits with keys outside the supplied frozen
IR can be preserved with `--allow-unlinked`; they remain visibly unlinked and
cannot enter replay training without an executable leaf program.

Make a disposable selection or migration from the IR, leaving the source
untouched. A tool map is a JSON object from canonical names to new names;
`null` removes a tool from the view. For example, mapping `end_turn` to `null`
and selecting `--empty-success-reply` adapts legacy terminal `done` to the
current reply-only completion convention.

```bash
printf '{"end_turn":null}\n' > runs/teacher-tool-map.json
.venv/bin/python scripts/project_teacher_trajectory_ir.py \
  runs/teacher-leaves-ir-backfill-s73-first8.trajectory-fresh.ir.jsonl \
  runs/teacher-leaves-ir-backfill-s73-first8.selected.ir.jsonl \
  --accepted-only --empty-success-reply --tool-map runs/teacher-tool-map.json
```

Projection does not itself admit action turns to SFT. The leaf output checks
validate the final text, not every choice. Before training on teacher actions,
replay them against the frozen leaf program and check intermediate states,
ordered effects, line marks, and errors; then render accepted structured turns
with the student's template. Keep rejected and failed trajectories for
contrastive analysis rather than treating them as successful demonstrations.

The legacy `scripts/materialize_teacher_trajectory_ir.py` performed that replay for
accepted, completed leaves. It reconstructs the exact frozen leaf request,
runs the selected decisions through the current harness, requires the same
final value, and compares recorded tool outcomes when present. It emits
template-neutral turns compatible with `scripts/export_sft.py`; the student's
chat template is applied only at export. Exposed teacher reasoning stays as
`teacher_reasoning` metadata on each turn and can be included or removed in a
later training view.

```bash
.venv/bin/python scripts/materialize_teacher_trajectory_ir.py \
  runs/teacher-leaves-ir-backfill-s73-first8.trajectory-fresh.ir.jsonl \
  runs/teacher-leaves-ir-backfill-s73-first8.replayed-turns.jsonl
```

The eight previously admitted seed-73 leaves replayed to 16 student turns
after the obsolete terminal `done` calls were migrated to empty final turns.
The four reply-only dry-run leaves also replayed to eight turns after their
missing final turns were explicitly reconstructed from successful outcomes.
Those legacy records have no captured reasoning; future audits retain whatever
reasoning field the server exposes. Self-review decisions are retained in the
source IR but are not yet rendered as student review targets by this leaf
materializer.
