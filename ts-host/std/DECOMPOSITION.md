# `@natlang/std`: decomposition (convergence and workspace families, WS3 S2)

`@natlang/std` is natlang's library of semantic combinators. Each combinator is TypeScript mechanism (exact
comparison, loops through `iterateOn`, folder transactions, record assembly) around named judgments. The judgments
that the library itself supplies are `.nl` functions in this folder; they ship as their exact bytes
(`scripts/generate-builtin.mjs` embeds them in `src/std/sources.generated.ts`), so Node, the npm package and the
browser bundle run the same functions. A program imports the combinators from `@natlang/node/std` (or
`@natlang/browser/std`) and the shipped judgments by name from the same module, so it can call, `decide`, `.in`,
`.on`, trace and specialise them like its own.

Values pass through by reference: a combinator never serialises, clones or reads a state it carries, so a
`Neuralese<T>` value moves through it unchanged and only the judgments that take it see it.

## Combinators and their parts

| Combinator | Part | Form | Why |
|---|---|---|---|
| `semanticRepair(value, { check, fix, attempts })` | run `check` on a value | crisp (the caller's exact checker) | exactness: compiler, schema or test output |
| | turn the problems into a fixed value | the caller's `fix` (usually inline `nl`) | the judgment belongs to the caller's domain |
| | loop: fix, re-check, stop on no problems or on a value seen before | crisp, through `iterateOn(...).withLimit` | an exact stopping predicate needs a bound the author chooses (`attempts`) |
| | answer at the bound | crisp: `{ ok: false }` with the value that had the fewest problems | a deliberate answer instead of an exception |
| `fixedPoint(step, initial, { settled?, maxSteps? })` | one revision | the caller's `step` | caller's domain |
| | has the revision stopped changing the meaning? | `settled.nl` (`readout: decision`) by default | the stop is a judgment; with it the loop needs no step bound (the progress judge watches it) |
| | exact settledness instead | the caller's `settled(previous, current)` plus `maxSteps` | when equality is exact, it is code |
| `semanticMerge(base, ours, theirs, options)` | three-way walk over records, unchanged and identical sides | crisp | exact structure |
| | what is each conflicting edit for? | `intentOf.nl` | judgment |
| | can both purposes be kept? | `compatible.nl` (`readout: decision`), gated on `floor` | finite judgment, calibrated |
| | write the value that keeps both | `combine.nl` | judgment |
| | is the combined value well-formed? | crisp: same kind as the base value, then the caller's `check` | exact verifier the model cannot edit |
| | what could not be merged | crisp: `unresolved[]` with both values, both intents and the reason; the merged value keeps the `keep` side there | explicit, never silent |
| `proposeAndVerify(folder, reducer, { verify, attempts, args })` | write a draft in a private copy | `folder.propose(reducer, ...args, problems)` | transaction mechanism |
| | judge the draft | the caller's exact `verify(proposal)` | a verifier is a service, never a helper the reducer can edit |
| | retry on the previous draft with the problems | crisp: the next proposal starts from the failed draft (`draft.select`), the loop is `semanticRepair` | one implementation of the repair loop |
| | install | `folder.accept(proposal)` only when `verify` reports no problems | exact |
| `branchAndSelect(folder, branches, { verify?, goal?, select? })` | each branch writes its own private copy | `folder.propose`, concurrently | transaction mechanism |
| | drop branches that fail | the caller's `verify` | exact |
| | choose among the passing branches | the caller's `select`, else `beats.nl` (`readout: decision`) pairwise over the changes against `goal`, else the first passing branch | judgment where a goal is stated |
| | install the chosen branch | `folder.accept` | exact |

## Traces

Each combinator run records one trace `std:<name>` under the call that ran it (the same `parentCallId` its own
model calls have), with one event per stage naming the call IDs that stage made. Its judgments and the caller's
functions are ordinary calls, so `natlang traces`, `natlang compilations` and the call store see each of them.
