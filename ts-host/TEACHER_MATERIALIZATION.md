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

The collector journals every exact model request and response before executing
the proposed actions. If collection is interrupted mid-program, the next run
replays those decisions through the fresh deterministic fixture runtime and
continues at the first request that was never answered. A request digest mismatch
fails closed. The journal is deleted when the job publishes its final accepted
or rejected row; only final rows enter materialization.
The Studio collector uses the same response-before-action journal and removes it
after publishing a case result.

The output contains one record for each teacher decision (which can contain one
or more actions) or checkpoint response. In addition to the lossless semantic
`decision` object, each record has the standard template-neutral
`messages`/`tools`/`target` fields consumed by `scripts/export_sft.py`.
Every record retains the source task, program IR, provenance, raw response hash,
teacher text and exposed reasoning. Offered function schemas are normalized to
`{name, description, parameters}`. Assistant calls retain their source tool name
and arguments, and each executed call links to the exact ordered action-ledger
event, including its trace sequence, result text, outcome, and diagnostics.
Unexecuted trailing calls and failed proposals remain visible in semantic IR but
receive `training_admission.approved: false`; the SFT exporter skips them. This
preserves teacher choices without teaching a rejected action as positive gold.
Materialization fails closed if action-ledger events remain unlinked.
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

Render approved decisions with the model server's selected chat template:

```sh
node scripts/export-native-sft.mjs data/teacher-native-turns.jsonl data/teacher-native.sft.jsonl \
  --server http://127.0.0.1:8081 --workers 4
```

This Node exporter preserves exposed reasoning and exact structured teacher
choices, skips denied decision-level admissions, checks template prefix and end
token boundaries, and records the template digest in its manifest.
