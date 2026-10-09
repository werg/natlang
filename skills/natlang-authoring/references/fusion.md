# Writing orchestrators that can fuse

Fusion is invisible in source: you never write `Neuralese` for it, and declared types stay as they are. What you control
is whether `natlang check` can prove that a stage's result is read only by the next stage. A hand-off is fusible when
the value goes from one stage into one other stage and nothing else touches it.

- Write the chain so the data flow is visible: `tree = parse(source)` then `analyze(tree)`, or `analyze(parse(source))`.
  Prose such as "parse the source, then analyze it" is a chain the check cannot prove, so it stays text.
- Do not read the middle value in the orchestrator when you do not need to: no `tree.errors`, no "when tree is empty",
  no storing it into another record, no passing it to a service or a crisp helper. Any of those makes the orchestrator a
  second reader.
- Use one stage's result for one next stage. A value sent to two stages is not fused.
- Keep both stages on the same `model:`.
- If a person or a later check should see the intermediate (a plan to approve, a diagnostic), say so in the orchestrator
  where it is used; the planner keeps such values as text.

`natlang check --fusion` lists every candidate hand-off with the reason it is or is not fusible. Details: `plans/FUSED_PIPELINES.md`.
