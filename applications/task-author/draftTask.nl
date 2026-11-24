---
description: Draft a Semantic-FP training task from a seed, with the answers that qualify its oracle.
args:
  seed: TaskSeed
  problems: string[]
  previous?: AuthoredTask
returns: AuthoredTask
---
Author a realistic Semantic-FP task for seed (using seed.construct and seed.oracleKind) where TypeScript handles
types, data and exact control flow while small typed `.nl` functions or inline `nl` calls handle the genuine semantic
judgments. Put `natlang.json`, `types.ts` and the program files in files, and include seed.construct in constructs.

Write three kinds of answers to the task. reference is a correct answer that the oracle must accept. negatives are at
least two plausible wrong answers that the oracle must reject. keywordBaseline is the best answer a rule matching
literal cue words in the inputs would give; the task is only worth training on when the oracle rejects it.

If previous is provided and problems is non-empty, refine previous to fix every listed issue while keeping seed.id and
seed.splitGroup.
