# Sem-Sheet (`applications/sem-sheet/`): Decomposition

Tier 1 Flagship A3 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3 & §3.5): browser-compatible analytical
spreadsheet combining deterministic DAG scheduling and crisp formulas in TypeScript with concurrent semantic columns,
tree-mode column reduction, `pluggable` column promotion, and `semanticMerge` collaborative editing.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `sheet.ts` — zero Node `fs` or `node:*` imports):**
  - Dependency graph extraction (`cellDependencies`, `expandRange`) and topological layer evaluation via
    `iterateOn(evaluateLayer, initial).withMeasure(remainingCells).until(state => state.pending.length === 0)`.
  - Pure recursive-descent formula evaluator (`evaluateCrispFormula`) supporting `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`,
    cell references, ranges, and arithmetic operators with zero `eval`.
  - Per-layer concurrent execution of semantic cells via `semanticMap`, `semanticFilter`, and tree-mode
    `semanticReduce(..., { mode: 'tree', combine })`.
  - Per-column promotion via `pluggable` (`crisp | nl | shadow`), emitting `pluggable_shadow` agreement events before
    promoting stabilized columns to zero-token crisp rules.
  - Collaborative workbook merging via `semanticMerge` with `checkMergedCellDefinition` syntax validation.

- **Typed semantic policy (`.nl` leaves):**
  - `evalSemanticCell.nl`: evaluates `=SEM(prompt, inputs)` cells with upstream text fenced as `Untrusted<string>`.
  - `classifyCell.nl` (`readout: decision`, returns `CellLabel = 'high' | 'medium' | 'low' | 'none'`): calibrated
    classification for finite-label columns.
  - `combineSummary.nl`: associative combiner for bottom-up tree reduction in `semanticReduce`.

## Safety & Browser Compatibility

- Core `sheet.ts` and `types.ts` import only `@natlang/node` and `@natlang/node/std` (zero Node-only built-ins),
  allowing direct execution in browser/wllama playgrounds as well as Node.
- All cell content passed to `.nl` leaves is wrapped with `untrusted(...)` and typed `Untrusted<string>`.
- Well-founded termination is guaranteed by `withMeasure(remainingCells)` plus explicit circular-dependency detection.
