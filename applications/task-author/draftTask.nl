---
description: Draft a Semantic-FP training task from a seed and repair any structural or baseline issues.
args:
  seed: TaskSeed
  problems: string[]
returns: AuthoredTask
---
Author a realistic Semantic-FP task for seed (using seed.construct and seed.oracleKind) where TypeScript handles types, data, and exact control flow while small typed `.nl` functions or inline `nl` calls handle genuine semantic judgments.
If problems is non-empty, fix every listed issue while preserving seed.id and seed.splitGroup.
Include `natlang.json`, `types.ts`, and the program files in `files`, and set `constructs` to include seed.construct.
