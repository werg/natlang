# Wiki: decomposition, part by part

Every part of the wiki, with a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model (rare).
- **crisp**: a TypeScript helper (plumbing with no decision).
- **host**: outside the stages (identity, shape, atomic publish, the VM).

The executors are small models, so each function spells its algorithm as numbered steps over named data.

## Policy

- **Natural language: every decision the wiki makes.** What an edit means, whether two edits conflict, how to merge
  them, whether the merge kept both intents, what the page's sections and links are, whether a link was renamed,
  which cell results survive an edit, and which cells run and in what order.
- **Crisp: lexing and exact commit.** Finding `# ` headings and `[[links]]`, diffing blocks by text, rewriting a
  link token already decided. The host checks transport identity, update coverage, block shape (including that a
  JavaScript cell compiles) and publishes a revision atomically.
- **State model.** The host takes a snapshot (page, updates, records), a stage returns data (a `MergeDraft`, a
  `Maintenance`, a `CellPlan`), and the host commits it in one bounded synchronous step that re-checks every
  invariant. Effects are data: link repairs are ordinary `WikiUpdate`s merged by the same pipeline; the cell plan is
  batches the host runs. Derived values form a DAG: outline and links come from the page, cell dependencies from
  the cell and the page, cell verdicts from dependencies and the delta. No while loops: the repair of a merged text
  is `iterateOn` with a measure and a step limit; the rest is finite.
- **Pluggable hot paths.** Two parts run on every update or merge and have two implementations behind one
  interface, chosen by `WikiSettings`:
  `changes` (per update) and `staleness` (per merge).

## Edit pipeline (`merge.nl`)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Transport identity: profile, base revision, duplicate IDs, ordering | host | `prepare` | Exact. |
| Group updates by block | inline | `merge` | Trivial. |
| Reconcile each block, in parallel | fn | `merge/block` | Blocks are independent. |
| Change summary: kind, intent, adds, removes | fn, **pluggable** | `block/changes` (`summarize.nl` or `exact.ts`) | Hot: once per update. Crisp reads only text. |
| Conflict detection per pair | fn, decision | `block/relate` | A finite judgment with a floor (below 0.6 counts as contradictory). |
| Trivial relations (noop/format, identical text) | inline | `block` | Exact, saves a call. |
| Merge planning: components of contradictions, covered updates, the combine set | fn | `block/plan` | A graph algorithm on small data. |
| Merge of prose | fn | `block/refine/prose` | The merge itself. |
| Merge of a cell's source, only when the combined meaning is clear | fn | `block/refine/code` | A different task: runnable source. |
| Verification against each intent | fn, decision | `block/refine/honors` | Run per intent in parallel. |
| Rewrite with the lost intents named, until none is lost | fn, `iterateOn` | `block/refine` | One step writes or rewrites, then checks. Limit 3. |
| Failure: keep base text, all carried updates become conflicts | inline | `block` | Conservative and exact. |
| Assemble blocks, accounted, unresolved | inline | `merge` | Mechanical. |
| Coverage, order, shape, compile check, conflict identity, publish | host | `publish` | Exact durability. |

## Page structure maintenance (`maintain.nl`)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Find heading blocks and `[[links]]` | crisp | `maintain/scan.ts` | Lexing. |
| Sections: ids, nesting, membership, problems | fn | `maintain/outline` | The derived outline. |
| Link resolution, rename detection | fn | `maintain/resolve` | Per link, in parallel. |
| Rewriting a link token | crisp | `maintain/retarget.ts` | Applies a decided map. |
| Repairs as updates | inline | `maintain` | Effects are data. |
| Block delta between revisions | crisp | `maintain/delta.ts` | Exact comparison. |
| Cell dependencies (blocks, cells, files) | fn | `maintain/staleness/judge/depends` | A reading judgment. |
| Staleness: DAG over dependencies | fn, **pluggable** | `maintain/staleness` (`judge.nl` or `exact.ts`) | Hot: after every merge. Crisp drops all results. |
| Keep a retired result only if fresh and its source hash is unchanged | host | `settle` | The commit. |

## Cell evaluation policy (`cells.nl`)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Refusals: no such cell, unresolved conflict, auto on natlang cell, fresh | inline | `cells` | Policy as spelled rules. |
| One run per cell, user over auto | inline | `cells` | Policy. |
| Dependencies | fn (used) | `depends` via `uses` | Shared with staleness. |
| Batches by dependency layer; cycles refused | inline | `cells` | Policy. |
| Running a batch, the VM, the natlang cell, time limits, stale-on-change | host | `schedule`, `runCell` | Mechanism. |

## Refinement candidates

Constraints now written as instructions or checked by hand that should become types
(`Is<T, "...">`, `Untrusted<T>`; plans/REFINEMENT_TYPES.md). Status: `adopted` rows are in code; `open` rows are not yet. `WikiUpdate.text` is `Untrusted<string>` in `types.ts`; `prepare` marks each update's text `wiki update <id>`.

| Slot | Proposed type | Status |
|---|---|---|
| `WikiUpdate.text` and any text read from a page or an update, shown to the model | `Untrusted<string>` | adopted |
| `Change.intent` | `Is<string, "one sentence saying what the block should say or do after the update">` | open |
| `Change.adds`, `Change.removes` | `Is<string[], "short phrases, each a statement the text has that the other lacks">` | open |
| `Change.text` | `Is<string, "the update's text, copied exactly">` | open |
| `PlanStep.update_ids` | `Is<string[], "IDs of changes to this block, each ID in exactly one step">` | open |
| `Merged.text` (prose) | `Is<string, "carries every add of the given changes and keeps every sentence none of them touched">` | open |
| `Merged.text` (cell) | `Is<string, "a complete function body that returns a value of the cell's return type on every path">` | open |
| `Merged` when `clear` is false | `Is<string, "equal to the block's base text">` | open |
| `BlockOutcome.accounted` | `Is<string[], "every update ID of the block once, sorted">` | open |
| `MergeDraft.blocks` | `Is<WikiBlock[], "the base blocks' IDs in the base order">` | open |
| `Conflict.alternatives` | `Is<string[], "the text of every update in the conflict, in update ID order">` | open |
| `Section.id` | `Is<string, "the lower-case slug of the title, with -2, -3 for repeats">` | open |
| `LinkStatus.resolves_to` | `Is<string \| null, "an existing section ID or block ID, null unless status is ok or renamed">` | open |
| `WikiUpdate` from `maintain` (repairs) | `Is<WikiUpdate, "changes only link targets">` | open |
| `CellRequest.input` for `origin: "auto"` | `Is<string, "taken from a recorded request">` | open |
| `CellPlan.batches` | `Is<CellRun[][], "no run reads the result of a run in the same or a later batch">` | open |
| `Dependency.cells` | `Is<string[], "IDs of cell blocks of this page">` | open |

Prompt guards removed from the old `reconcile.nl`: "return runnable source only if the combined meaning is clear;
otherwise keep the old source and record a conflict" is now the `clear` flag of `Merged` (refinement above), and
"account for every update ID exactly once" is the `accounted` refinement; the host still checks both.
