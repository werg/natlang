# Teacher execution setup

The teacher is Ternary Bonsai 2 27B PTQ1_0, served by llama-server on port
8081. The probes record the actual `/v1/models` response, program, prompt,
tool transcript, runtime outcome, effects, and audit results. Teacher probes
are quarantined under `runs/`; they never append training references.

## Keep the execution path simple

- Use `natlang/prompts/tools_teacher_compact.md`, temperature 0, low reasoning
  effort, and a 256-token thinking budget.
- Use `LlamaServerDecoder(json_text_values=True, tool_aliases={"call":
  "call_function"})` for this teacher/server combination. Text values are plain
  strings; other values are JSON text. Runtime validation still decides
  whether a proposal fits its destination. Invalid proposals remain possible.
- Keep compact state, caller validation feedback, and no self-review or local
  repair loop. Expanded state and review prompts remain student experiments.
- A normal assistant reply signals successful completion. The harness checks
  return validity and closed numbered lines at that boundary. `report_error`
  and `report_blocker` remain failure signals.
- Existing values can be copied with `write(source=...)`. The runtime preserves
  types, permits one incidental JSON quoting layer when necessary, and never
  unwraps a `{ "value": ... }` object. Fold/iterate literal initializers follow
  the same quote rule; source references retain their existing behavior.
- Missing optional inputs are explicitly identified both in the opening state
  and on direct reads. An actual empty string remains a valid supplied value.

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
The resumed pass uses `--turn-tokens 1600`, started with 387 missing keys,
and writes `runs/teacher-leaves-frozen-s73-extended-20260919.jsonl`. The
original 700-token cap caused six responses to stop at `finish_reason=length`;
four responses in the resumed audit also hit the 1600-token cap. The active
watcher (`runs/finalize-synthetic-backfill-uncapped.log`) therefore runs one
more pass over missing keys with no separate per-turn cap, writing
`runs/teacher-leaves-frozen-s73-uncapped-retry-20260919.jsonl`, before it
creates `synthetic-all-after-uncapped-retry-<bank hash>` artifacts. All pass
audits remain available; admitted references are shared through
`data/leaf_references.jsonl`.

## End-of-turn completion

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

For legacy student trajectories, run `scripts/normalize_terminal_done.py SRC DST`
before `scripts/export_sft.py`. The normalizer removes the obsolete tool from
each turn, changes a standalone terminal target to an empty final turn, and retains
`done=N` line marks. It refuses mixed batches rather than guessing how to
split a turn. The source corpus is left intact.
