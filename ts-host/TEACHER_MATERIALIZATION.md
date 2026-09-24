# Native teacher trajectory materialization

`src/teacher/native-materializer.ts` converts accepted
`natlang.teacher_trajectory.native/1` collector rows into JSONL decision records
using `natlang.teacher_training_turn.native/1`. It runs entirely in Node and
preserves the captured teacher choices, reasoning, tool calls, and observations
in the native training records.

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
or more actions). In addition to the lossless semantic
`decision` object, each record has the standard template-neutral
`messages`/`tools`/`target` fields consumed by `scripts/export-native-sft.mjs`.
Every record retains the source task, program IR, provenance, raw response hash,
teacher text and exposed reasoning. Offered function schemas are normalized to
`{name, description, parameters}`. Assistant calls retain their source tool name
and arguments, and each executed call links to the exact ordered action-ledger
event, including its trace sequence, result text, outcome, and diagnostics.
Unexecuted trailing calls and failed proposals remain visible in semantic IR but
receive `training_admission.approved: false`; the SFT exporter skips them. This
preserves teacher choices without teaching a rejected action as positive gold.
The runtime also tolerates the narrow, semantics-free `const x = x` self-alias
mistake for an injected binding so the enclosing program can continue, records
`coerced-redundant-self-alias` on that action, and denies that decision positive
SFT admission.
Materialization fails closed if action-ledger events remain unlinked.
Rows without `outcome.accepted === true` are counted and skipped.

Each decision copies the messages from its captured native request and normalizes
assistant/tool messages into semantic calls and results. It does not build a
transcript by concatenating request histories.

The agent keeps each call's conversation within a context budget (`--context-tokens`,
default 16,384, the training length) by deterministic compaction: past three quarters of
the budget, the oldest tool outputs, then the oldest eval code, are replaced by fixed
stubs until the prompt is under half. Every decision records its exact request, so
compacted contexts are trained as the model saw them. Rows collected with the retired
conversation rollover (checkpoint turns) are rejected.

Render approved decisions with the model server's selected chat template:

```sh
node scripts/export-native-sft.mjs data/teacher-native-turns.jsonl data/teacher-native.sft.jsonl \
  --server http://127.0.0.1:8081 --workers 4
```

This Node exporter preserves exposed reasoning and exact structured teacher
choices, skips denied decision-level admissions, checks template prefix and end
token boundaries, and records the template digest in its manifest.
