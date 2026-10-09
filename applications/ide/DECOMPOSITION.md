# IDE: decomposition, part by part

Status: implemented on main (2026-10-09, owner: finish and integrate); review after the fact in plans/OWNER_REVIEW.md. Question 1 (canonical scenario comparison) is done; questions 2 and 3 stay open. `proposeEdit` takes a `feedback` argument for the one retry; the anchor and file checks are plain crisp checks in `locate` and `requestEdit` (no `Is<>` judge calls).

`IdeWorkbench` (`index.ts:23-114`) holds revisioned sources, checks, pinned runs, trace inspection and scenarios.
All of it is exact and stays crisp. Two natural-language functions connect it to a person:

- `interpret.nl` turns a request into an `EditPatch` with character offsets. A small model cannot count characters
  reliably; the instruction asks it to "read the whole relevant function before choosing offsets"
  (`interpret.nl:8-10`), and a wrong offset becomes a rejected or silently wrong edit (`index.ts:50-52`).
- `describe.nl` asks the model to write a whole editor view, including the source panel text (a copy) and the
  diagnostics (a copy), plus the explanation (`describe.nl:8-10`).

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **pluggable**.

## Policy

- **Natural language: what to change and what a trace event means.** Which file, which text to replace and with
  what, a concise reading of diagnostics, a plain-language reading of one recorded event.
- **Crisp: everything with an offset, a revision or a hash.** `locate` finds the anchor text and computes offsets;
  `edit` checks the range and the expected revision (`index.ts:47-57`); `check`, `run`, `inspect`, scenarios and
  `render` are exact (`index.ts:59-111`).
- **The model quotes; the host measures.** The model names an exact span of the current source (the anchor) and its
  replacement. Crisp finds the anchor's unique occurrence and derives `start`, `end` and `expected_revision`. When
  the anchor occurs zero times or several times, the count returns to the stage once as feedback.
- **The view's fixed parts are built by crisp.** Panels for the source and the raw diagnostics are copies of host
  values; the model writes only the two readings.
- **State model.** `requestEdit` is decide-then-commit: snapshot (revision, files), `proposeEdit` decides, `locate`
  derives, `edit` commits or reports `stale`/`rejected`. Derived values form a DAG: snapshot, file, anchor, patch,
  revision, check.

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Source revisions by hash, snapshot | crisp | `index.ts:30-45` | Exact. |
| Which file does the request concern | fn, decision | `edit/chooseFile` | Semantic: reads names and the request. |
| Which text to replace and with what | fn | `edit/proposeEdit` | The semantic core: an anchor, a placement and a replacement. |
| Offsets from the anchor | crisp | `edit/locate` | Counting characters is exact work; replaces "Give zero-based start and end offsets" (`interpret.nl:8-9`). |
| `expected_revision` | crisp | `edit/locate` | Replaces "carry the snapshot revision as expected_revision" (`interpret.nl:9-10`). |
| Edit: range check, stale revision check, new revision | crisp | `edit`, `index.ts:47-57` | Exact commit. |
| Validate a revision's project | service | `check`, `index.ts:59-64`, `validatePlaygroundProject` | The language service. |
| Run a pinned revision, retain trace | service | `run`, `index.ts:66-76` | The runtime. |
| Inspect one trace event, read-only | crisp | `inspect`, `index.ts:78-83` | Replaces "Never treat trace inspection as execution or replay" (`describe.nl:10`): `inspect` has no execute path. |
| Scenario add, evaluate | crisp | `index.ts:85-104` | Exact. See question 1 about the comparison. |
| Source panel | crisp | `view/assemble` | Copy of the snapshot. |
| Diagnostic panel (raw) | crisp | `view/assemble` | Copy of the check detail. |
| Readable summary of the diagnostics | fn | `view/summarizeDiagnostics` | Concise reading for a person. |
| Plain-language reading of one trace event | fn | `view/explainEvent` | Semantic. |
| Render to escaped HTML | crisp | `render`, `index.ts:106-111` | Exact escaping. |
| Event log | host | `drainEvents`, `index.ts:113` | Mechanism. |

## Natural-language functions, step by step

### `edit/chooseFile`

```
args: request: string, names: string[]
returns: Is<string, "one of names">
```

1. Read the request; list any file name or module name it mentions.
2. When one of `names` matches a mention, return it.
3. Otherwise return the name whose role (from its name and extension) fits the request best.

### `edit/proposeEdit`

```
args: request: string, name: string, source: string
returns: { anchor: Is<string, "occurs exactly once in source">,
           placement: "replace" | "insert-before" | "insert-after",
           text: string }
```

1. Read `source` and find the lines the request concerns.
2. Choose `anchor`: the smallest exact span of `source` (copied, including whitespace) that identifies the place.
   Extend it by neighbouring text until it identifies exactly one place.
3. Choose `placement`: "replace" when the anchor itself changes; "insert-before" or "insert-after" when text is
   added next to it.
4. Write `text`: the replacement or the inserted text, with the same indentation as its surroundings.

### `edit/locate` (crisp)

1. Count occurrences of `anchor` in `source`. When the count is not 1, return the count as feedback to
   `proposeEdit` (once); a second failure is reported as a rejected edit.
2. For "replace": `start` is the anchor's index, `end` is `start + anchor.length`. For "insert-before": `start = end`
   at the index. For "insert-after": `start = end` at `index + anchor.length`.
3. Return `{ name, start, end, text, expected_revision: snapshot.revision }`.

### `view/summarizeDiagnostics`

```
args: checked: CheckReport
returns: Is<string, "at most five lines, one per distinct diagnostic">
```

1. When `checked.status` is `checked`, return "no errors".
2. Group diagnostics by file and message; write one line per group: file, what is wrong, where.

### `view/explainEvent`

```
args: event: TraceView   // event_json is Untrusted<string>
returns: string
```

1. Read the event kind and fields in `event_json`.
2. Say what the event records (a call, a result, a model turn, an error) and its position `index` of `total`.
3. Quote the values that matter, as recorded.

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `chooseFile` result | `Is<string, "one of names">` | crisp |
| `proposeEdit.anchor` | `Is<string, "occurs exactly once in source">` | crisp; feeds the repair loop with the count |
| `EditPatch.expected_revision` | host-filled | crisp |
| `EditPatch` | `Is<EditPatch, "0 <= start <= end <= source length of name">` | crisp (`index.ts:50-51`) |
| `TraceView.event_json` | `Untrusted<string>` | crisp marking (events carry model and tool text) |
| `summarizeDiagnostics` result | `Is<string, "at most five lines, one per distinct diagnostic">` | crisp for line count; judged for "distinct" |
| `EditorView.panels` | `Is<ViewPanel[], "source, diagnostics, trace panels in that order">` | crisp (built by the assembler) |
| `RunReport.status` | closed union (already) | crisp |

## Model-facing changes needing live measurement

1. **Anchor edits replace offset edits.** Measure patch correctness (applied edit equals the intended edit) on the
   requests in `ts-host/test/ide-workbench.test.mjs` and a set of multi-line, repeated-text cases. Hypothesis: the
   rate of stale and rejected edits drops to the anchor-uniqueness failures only.
2. **`interpret.nl` split** into `chooseFile` and `proposeEdit`.
3. **`describe.nl` split** into `summarizeDiagnostics` and `explainEvent`; the view layout and source panel are
   assembled by crisp. The sentence "Never treat trace inspection as execution or replay" is removed.
4. **Retry text** for an anchor count other than one ("the anchor occurs N times; extend it with the neighbouring
   text").
5. **`Untrusted<string>` rendering** of trace event text.

## Questions for the owner

1. Scenario evaluation compares `run.value_text === JSON.stringify(scenario.expected)` (`index.ts:101`). Key order
   changes the string. Use the canonical value comparison from the runtime (`canonicalValue`) instead; this is a
   crisp fix with no model-facing change.
2. A scenario failure currently reports `matches: false`. Add an advisory natural-language `explainMismatch` on the
   same pattern as the failure-explanation program? Not included in this draft.
3. `proposeEdit` edits one span per request. Multi-span requests ("rename X everywhere") can be a list of anchors;
   whether to allow that depends on how often requests need it.
