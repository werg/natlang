# sem-sheet

Tier 1 Flagship A3: browser-compatible reactive spreadsheet mixing crisp formulas with concurrent semantic columns,
calibrated classification columns (`readout: decision`), tree-mode column summaries, `pluggable` column promotion, and
`semanticMerge` collaborative editing.

```sh
natlang check applications/sem-sheet
```

## Architecture

- **Browser-compatible core**: zero `node:*` imports in `sheet.ts` or `types.ts`.
- **Topological DAG (`iterateOn` + `withMeasure`)**: `evaluateSheet` steps layer-by-layer with `remainingCells` as its
  strictly decreasing measure, running crisp formulas synchronously and semantic cells concurrently per ready layer.
- **Semantic combinators**: `semanticMap` fans out `=SEM` and `classify` cells; `semanticFilter` evaluates
  `=SEM_FILTER` cells; `semanticReduce(..., { mode: 'tree' })` rolls up ranges via `combineSummary.nl`.
- **Column promotion (`pluggable`)**: stabilized classification columns run in `shadow` mode (emitting
  `pluggable_shadow` agreement telemetry) before promotion to `crisp` (zero LLM calls).
- **Collaborative merge (`semanticMerge`)**: `mergeSheetDiffs` merges disjoint cell edits crisply and resolves
  same-cell conflicts by intent compatibility plus exact formula syntax checks.
