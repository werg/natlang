# P10 — Natlang notebook with data operations

Status: proposed implementation. [Shared capabilities](README.md).

## Natlang prerequisites

C3 runs cell source supplied as data with explicit inputs. C1 binds required evaluators, initially TS and SQL; engine selection is explicit where snippets are generated. C7 converts results into the same natlang value model. C4 records cell dependencies and execution evidence. C5 drives interactive use later; C6 only accelerates proven independent cells.

## Programme and typed boundary

`step.nl(state, event) -> NotebookState`, `run_cell.nl(cell, inputs) -> CellResult`, `interpret_table.nl` and `explain_result.nl`. Exact helpers validate declared dependency graphs and compute invalidation. Natlang may propose dependencies but the committed bindings are explicit before execution.

Cell records hold source revision, engine requirements, input/output names, dependency revisions and outcome. Table values are bounded summaries/row samples or ordinary IDs for native tables. Keep connections, cursors and large buffers inside eval. SQL NULL, missing columns and numeric conversions must follow C7, not engine-specific guesses.

## Crisp environment

A notebook-scoped environment holds datasets and selected database snapshots. Illustrative meta-APIs load/check/run cell programmes and query tables. A TS snippet can directly use a native table object; a SQL engine receives a specified connection/view and query bindings. It does not access a TS object simply because both engines share the word “table.” Cross-engine data sharing is an explicit host mapping or materialisation step.

Separate pure, declared-input cells from cells intentionally using retained mutable environment state. The latter are supported, but their hidden dependencies prevent automatic cache/invalidation claims unless the host records their read/write sets or conservatively invalidates. Mark this in the UI and trace.

## Reduction and stream shape

Start with sequential cell runs in dependency order. The notebook application's Fold handles edits, run requests and completions. Each output is associated with source and input revisions. A result from an older revision remains historical but cannot mark current descendants fresh. Stream progress is optional; finite results retain ordinary cell semantics.

## Delivery and checks

1. Load two CSV fixtures, semantically map labels, join/aggregate exactly and explain output.
2. Introduce explicit TS/SQL engine selection and incompatible-result tests. Gate: same boundary validation and recorded engine identity.
3. Add source/input edits and invalidation. Gate: exported notebook bundle reproduces declared-input results without depending on execution order accidents.
4. Add retained-state cells and background UI. Gate: uncertainty about dependencies/replay is visible, not hidden behind a green cache indicator.

Test cell cycles, missing bindings, null/empty differences, large results, query failure, stale completions and duplicate effectful execution.

## Trace and teacher

Link cell run traces through parent notebook IDs and dependency revisions. Record actual query observations or restorable dataset snapshots for replay. Teacher cases cover selecting exact computation, semantic munging, correct engine use and acknowledging incomplete data. No universal notebook variable namespace is added to natlang; cell interoperability is explicit binding through the embedding.
