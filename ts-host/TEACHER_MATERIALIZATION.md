# Native teacher trajectory materialization

`src/teacher/native-materializer.ts` converts accepted
`natlang.teacher_trajectory.native/1` collector rows into JSONL decision records
using `natlang.teacher_training_turn.native/1`. It runs entirely in Node and
does not replay examples through the legacy Python runtime.

Build the TypeScript host, then materialize a collector merge:

```sh
npm run build:node
node scripts/materialize-native-teacher.mjs runs/teacher/merged.jsonl data/teacher-native-turns.jsonl
```

Pass `--replace` when rebuilding a derived view from resumable source jobs.

The output contains one record for each teacher decision (which can contain one
or more actions) or checkpoint response. In addition to the lossless semantic
`decision` object, each record has the standard template-neutral
`messages`/`tools`/`target` fields consumed by `scripts/export_sft.py`.
Every record retains the source task, program IR, provenance, raw response hash,
teacher text and exposed reasoning. Offered function schemas are normalized to
`{name, description, parameters}`. Assistant calls retain their source tool name
and arguments, and each executed call links to the exact ordered action-ledger
event, including its trace sequence, result text, outcome, and diagnostics.
Materialization fails closed if an accepted call cannot be linked to the ledger.
Rows without `outcome.accepted === true` are counted and skipped.

Each decision copies the messages from its captured native request and normalizes
assistant/tool messages into semantic calls and results. It does not build a
transcript by concatenating request histories. A checkpoint is retained as a
teacher decision with its note; the following decision starts a new segment and
uses the fresh system/user opening captured by the collector. This prevents an
earlier continuation's dialogue from being pasted into the later segment.

The native collector controls trajectory length with turn and message item
limits. Materialization preserves those snapshots without token-based truncation,
so long code or reasoning remains intact. To make examples shorter, lower the
collector's `segment_turns` or `segment_messages`; those bounds count decisions
and messages rather than imposing a completion-token cap.
