# fusion-planner: which hand-offs stay Neuralese

A natlang program (plans/FUSED_PIPELINES.md). The runtime's crisp fact service lists the candidate edges of a program
with their readers, models and types. This program decides each edge, `fuse` or `keep-text`, with a reason. A crisp
verifier checks the answer against the facts before the runtime uses it, so this program can only make the plan more
conservative than the facts allow, never less safe.

```
planner.nl               orchestrator: one decision per edge, all at once
planner/decide.nl        the numbered conditions for one edge and the final answer
planner/needsReading.nl  the one judgment: does a person or a later check need to read the value as text
```

The runtime loads it when `fusion.planner` in natlang.json is `nl` or `shadow`. Crisp (`fusion.planner: crisp`) is the
default and needs no model.
