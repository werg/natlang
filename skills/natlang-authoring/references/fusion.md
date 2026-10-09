# Writing orchestrators that can fuse

Fusion is invisible in source: you never write `Neuralese` for it, and declared types stay as they are. What you control
is whether `natlang check` can prove that a stage's result is read only by the next stage. A hand-off is fusible when
the value goes from one stage into one other stage and nothing else touches it.

## The explicit chain

The language already has the form that states a chain exactly: **a call inside a call**, or **a name that only the next
stage uses**. Nothing new is needed.

```
checked = analyze(parse(source))        // parse's result goes into analyze and nowhere else
checked = analyze(tree)                 // after `tree = parse(source)`, and tree appears nowhere else
declare(analyze(parse(source)))         // a chain of three: both inner hand-offs are exact
```

In prose, in a numbered step of the orchestrator, write the call with its argument inside it
(`statement = plan(parse(request, kind, catalog, today), catalog)`) and say what the stages do after it. The check reads
`B(A(x))` as: the value of `A(x)` is B's argument and has no name, so no one else can read it. It reads
`v = A(x)` as: every later mention of `v` in a call or a sentence is a reader, so `v` must appear once, as an argument of B.

The same holds in TypeScript orchestrators (`analyze(await parse(source))`, or `const tree = await parse(source)` used
only as `analyze(tree)`), where the proof is exact syntax, and in the code the orchestrating model writes at run time.

Prose such as "parse the source, then analyze it" is a chain the check cannot prove, so it stays text.

## Give the next stage what the orchestrator needs from the middle

An orchestrator that reads the middle value (`tree.errors`, "when tree is empty", a record built from it, a log line, a
service call, a crisp helper) is a second reader, and the hand-off stays text. Instead of reading it, make the consumer
carry what you need forward and read it from the consumer's result:

- Before: `parse(source)` and `analyze(syntax)`, and "if either reports diagnostics, stop". The orchestrator reads both.
- After: `checked = analyze(parse(source))`, and `analyze` starts its diagnostics with the syntax's own ("when syntax has
  diagnostics, return them unchanged, with no declarations"). The orchestrator reads only `checked.diagnostics`, so parse to
  analyze fuses. The value `checked` is read by the orchestrator, so analyze to declare stays text, which is correct.

Do this only when it is faithful: the consumer's result must hold what the orchestrator needs, and the orchestrator
must behave the same when the consumer stops early. Where a plan decides the chain at run time (nldb's executor runs the
steps of whatever plan the optimizer wrote), no static chain exists to write; those hand-offs stay text, or are proven by
observation (below).

## Rules of thumb

- Use one stage's result for one next stage. A value sent to two stages is not fused.
- Do not read the middle value in the orchestrator when you do not need to: no field access, no condition on it, no
  storing it into another record, no passing it to a service or a crisp helper. Any of those makes the orchestrator a
  second reader.
- Keep both stages on the same `model:`.
- Do not export a variable that holds a stage result from a TypeScript module: importers read it.
- If a person or a later check should see the intermediate (a plan to approve, a diagnostic), say so in the orchestrator
  where it is used; the planner keeps such values as text.

## Observed readers

A chain the prose only implies can still be proven from what orchestrating models actually ran: with
`"fusion": { "observed": true }` (or `{ "minRuns": 20, "store": "path" }`) in natlang.json, `natlang check` also reads the eval
code recorded in the call store for the orchestrator as it is now. A hand-off fuses on that evidence only when at least
`minRuns` model-driven runs passed the value from the first stage to the second and to nothing else, and no run read it
anywhere else. The plan records the run count and the store revision; `natlang check --fusion` shows `readers observed`.
Observation is statistical: use `shadow` mode before `on` for such edges.

`natlang check --fusion` lists every candidate hand-off with the reason it is or is not fusible. Details: `plans/FUSED_PIPELINES.md`.
