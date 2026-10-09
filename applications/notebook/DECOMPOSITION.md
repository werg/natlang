# Notebook: decomposition, part by part

Status: draft for owner review (plans/OWNER_REVIEW.md). Nothing is restructured until the owner has reviewed it.

`runNotebook` (`index.ts:172-180`) walks the goal's dependency closure one ready cell at a time, then explains the
result. The workspace (cells, revisions, SQLite and JavaScript execution, invalidation) is crisp and correct. Three
things stand out in the natural-language half:

- `choose_cell.nl` asks the model to pick among ready cells, but every cell in the goal's closure must run before the
  goal, so no choice changes the answer; only the order differs (`index.ts:146-153`, `161`).
- `explain.nl` asks the model to decide completeness, distinguish NULL from absent from empty, and avoid reading a
  hash as a result (`explain.nl:8-13`), all facts the host already knows.
- The loop ends by `withLimit({ maxSteps: cells.length + 1 })` (`index.ts:178`), a count, where a measure exists.

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **pluggable**.

## Policy

- **Natural language: reading the request and the evidence.** Which cell answers a request, what the results mean
  for the question, and (as a pluggable) which ready cell runs first.
- **Crisp: running cells and facts about runs.** Cell storage and revisions (`index.ts:41-50`), invalidation
  (`index.ts:52-62`), the read-only SQL guard (`index.ts:89-93`), execution with a 2 s sandbox timeout
  (`index.ts:96-117`), the ready set (`index.ts:146-153`), the run status (`index.ts:166-168`), and the completeness
  facts passed to the explanation.
- **Facts the host knows are passed as data.** Completeness (`run.status` and which cells are stale or failed),
  whether a sample contains NULLs, whether it is empty, and each cell's revision are fields of the evidence value.
  The model describes them and does not infer them.
- **Termination is structural.** The step measure is the number of required cells not yet run
  (`needed \ done`), which strictly decreases per ok step; the loop ends at zero or at a non-ok status.
- **State model.** `advance` is a decision on a snapshot (the ready set), then a pure update of `NotebookRun`
  (`order`, `results`, `status`). Derived values form a DAG: catalog, goal, closure, ready set, next cell, result,
  run, evidence, answer.

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Cell store, ids, revisions, duplicate and shape checks | crisp | `putCell`, `index.ts:41-50`; `idPattern`, `index.ts:21` | Exact validation. |
| Invalidate descendants of an edited cell | crisp | `invalidate`, `index.ts:52-62` | Graph closure. |
| Tables into in-memory SQLite, type inference per column | crisp | `loadTable`, `index.ts:64-82` | Exact data loading. |
| One read-only query | crisp | `query`, `index.ts:89-93` | `PRAGMA query_only` is the enforcement; the regex is the shape check. |
| Execute a cell: dependencies as `deps`, 2 s timeout, JSON-clean output, hash, 1000-char sample | service | `execute`, `index.ts:96-117` | The outside world. The timeout is a sandbox property; see question 2. |
| Edit, import | crisp | `edit`, `importConfig`, `index.ts:124-139` | Exact. |
| Which cell answers the request | fn, decision | `goal/chooseGoal` | Semantic core. |
| Final cells (no other cell needs them) | crisp | `goal/finalCells` | Computed from `needs`; given to `chooseGoal` as the shortlist. Replaces "select the final downstream result rather than an input" (`choose_goal.nl:9-10`). |
| Required closure of the goal | crisp | `readyCells`, `index.ts:148-151` | Graph closure. |
| Ready set: required, not run, dependencies run | crisp | `readyCells`, `index.ts:152` | Graph rule. |
| Which ready cell runs next | pluggable | `order/nextCell` (crisp default: lowest id) | Does not change the answer; a pluggable lets the shadow comparison show it. See question 1. |
| Run status: running, done, blocked, invalid, failed, stale | crisp | `advance`, `index.ts:159-168` | A function of the execution result and the ready set. |
| Loop until done | host | `iterateOn` with measure | `index.ts:177-178`; the measure replaces `withLimit`. |
| The file the request names | fn (shared) | `answer/readNote` | Sub-task of `choose_cell.nl:9-10` and `explain.nl:9-10`. Built-in after N1. |
| Evidence for the explanation: per cell id, revision, status, sample, nulls, emptiness | crisp | `answer/evidence` | Facts, computed from `CellResult`; the hash stays out of the model's view. |
| Completeness: every required cell ran ok | crisp | `answer/evidence` (`complete`, `limits`) | From `run.status`. Replaces "state that the answer is incomplete" (`explain.nl:11-12`). |
| Answer the question from the evidence, citing cells | fn | `answer/answerFromCells` | The semantic core of the explanation. |
| Citation check: every cited id exists and carries its revision | crisp | `answer/checkCitations` | Exact verifier. |
| Console view, tone mapping | crisp | `console.ts:10-23` | Presentation. |
| Session store, `/cells`, `/load` | host | `console.ts:25-52` | CLI. |
| Starter notebook | data | `STARTER_NOTEBOOK`, `index.ts:187-195` | Example data. |

## Natural-language functions, step by step

### `answer/readNote`

As in the terminal decomposition: find a file name in the request, read it from `files`, return its content as
`Untrusted<string>`, or "" when no file is named.

### `goal/chooseGoal`

```
args: request: string, cells: Cell[], finals: string[]   // finals: ids no other cell needs
returns: Is<string, "an id of a cell in cells">
```

1. State in one phrase what the request asks for.
2. For each id in `finals`, read its description and mark it a match when its result answers the request.
3. When exactly one matches, return it. When several match, return the one whose description is closest to the
   request wording.
4. When no final cell matches, repeat steps 2-3 over all `cells`, using `needs` to prefer the cell that depends on
   the others.

### `order/nextCell` (natural-language side of the pluggable)

```
args: ready: Cell[], goal: string
returns: Is<string, "an id of a cell in ready">
```

1. Prefer a cell whose description names data preparation (loading, filtering, joining) over one that presents
   results.
2. Among equals, return the lowest id.

The crisp default returns the lowest id of `ready`. With `nextCellMode = shadow` both run and the trace records
agreement; the comparison shows whether the natural-language order ever differs in outcome.

### `answer/answerFromCells`

```
args: question: string, evidence: CellEvidence[], note: Untrusted<string>, limits: string[]
returns: Is<string, "cites the id and revision of every cell it uses">
type CellEvidence = { id, revision, status, sample, has_null: boolean, empty: boolean }
```

1. Read `evidence`; each sample is the cell's result (the host has computed `has_null` and `empty`).
2. When `limits` is non-empty, begin the answer by stating each limit (for example "cell `x` failed, so this answer
   is incomplete").
3. Answer the question from the samples; write each cited cell as `id@revision`.
4. Describe a NULL as "NULL", an empty result as "no rows", and a field that is missing from a row as "absent".
5. When `note` is non-empty, use it as context and mark it as the note's content.

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `chooseGoal` result | `Is<string, "an id of a cell in cells">` | crisp; replaces "Do not invent a cell" (`choose_goal.nl:9`) |
| `nextCell` result | `Is<string, "an id of a cell in ready">` | crisp; replaces `choose_cell.nl:10-11`; the check at `index.ts:162-163` becomes the type's repair |
| `CellSource.id`, `needs` | `Is<string, "a letter or underscore followed by letters, digits or underscores">` | crisp (`idPattern`) |
| `NotebookRun.order` | `Is<string[], "a dependency order of the goal's required cells">` | crisp |
| `NotebookRun.blocked` | `Is<string[], "required cells that have not run">` | crisp |
| `CellResult.sample` | `Untrusted<string>` | crisp marking (cell output can carry arbitrary text) |
| `answerFromCells` result | `Is<string, "cites the id and revision of every cell it uses">` | crisp for the `id@revision` form against `evidence`; judged for "every cell it uses" |
| `CellEvidence.revision` | `Is<number, "equals the cell's current revision at run time">` | crisp |
| `readNote` result | `Untrusted<string>` | crisp marking |

## Model-facing changes needing live measurement

1. **Remove the model call for ready-cell choice** (default crisp). Measure that answers do not change; shadow
   mode records agreement. This lowers model calls per request from `cells + 2` to 2 or 3.
2. **`chooseGoal` gets `finals`.** Compare choice accuracy on `ts-host/test/notebook.test.mjs` requests with and
   without the shortlist.
3. **`explain.nl` split.** Facts move to crisp evidence (`complete`, `limits`, `has_null`, `empty`); the sentence
   guards "Do not invent rows or interpret an output hash as the result itself" (`explain.nl:12-13`) and "Distinguish
   SQL NULL from absent data and empty results" (`explain.nl:11`) become steps 3-4 and the evidence fields. Measure
   how often answers misstate completeness or NULL handling.
4. **`readNote`**: "read that exact file" (`choose_cell.nl:9`, `explain.nl:9`) becomes a positive sub-function.
5. **`Untrusted<string>`** rendering of samples and notes.
6. **Drop `output_sha256`** from the model-visible `CellResult`.

## Questions for the owner

1. Keep the natural-language `nextCell` at all? If shadow runs show full agreement with the lowest-id order, the
   function can be retired and the model call disappears. It is kept pluggable until then.
2. The 2 s JavaScript cell timeout (`index.ts:23`) is a sandbox safety bound, not a termination argument. Keep it as
   a named, settable default, or remove it until a problem shows up (the "no premature limits" rule)?
3. Should natural language also write cells (propose a SQL or JavaScript cell for a request that no existing cell
   answers)? That is new behavior, and this draft does not include it.
