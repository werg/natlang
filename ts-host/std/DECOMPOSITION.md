# `@natlang/std`: decomposition (collections, convergence, policy and workspace families)

`@natlang/std` is natlang's library of semantic combinators. Each combinator is TypeScript mechanism (bounded
concurrency, exact grouping, blocking, Copeland ranking, loops through `iterateOn`, folder transactions, record
assembly) around named judgments. The judgments that the library itself supplies are `.nl` functions in this folder;
they ship as their exact bytes (`scripts/generate-builtin.mjs` embeds them in `src/std/sources.generated.ts`), so Node,
the npm package and the browser bundle run the same functions. A program imports the combinators from
`@natlang/node/std` (or `@natlang/browser/std`) and the shipped judgments by name from the same module, so it can call,
`decide`, `.in`, `.on`, trace and specialise them like its own.

Values pass through by reference: a combinator never serialises, clones or reads a state it carries, so a
`Neuralese<T>` value moves through it unchanged and only the judgments that take it see it.

## Combinators and their parts

| Family | Combinator | Exact mechanism (TS) | Semantic judgment (`.nl` / caller) |
|---|---|---|---|
| Collections | `semanticMap(items, fn, { concurrency?, carry?, view? })` | bounded concurrent fan-out, order preservation, optional per-item Neuralese view (`carry: 'neuralese'`) | caller's `fn` (plus the builtin `view` when `carry: 'neuralese'`) |
| | `semanticFilter(items, predOrCriterion, { floor?, concurrency? })` | concurrent fan-out, confidence threshold, exact array filter | caller's predicate or shipped `meets.nl` (`readout: decision`) |
| | `semanticPartition(items, classifier, { floor?, concurrency? })` | concurrent `decide`, exact grouping into `groups` + `uncertain` | caller's finite classifier |
| | `semanticReduce(items, initial, step, { mode?, combine?, concurrency?, carry?, view?, read? })` | left fold (`linear`) or parallel balanced pair reduction (`tree`), optional soft-state carry + final `read` | caller's `step` (and `combine`), carrying `S \| Neuralese<S>` by reference or through the builtin `view` |
| | `semanticMatch(left, right, { relation?, match?, block?, floor? })` | exact `block` pruning, unmatched set calculation | caller's `match` or shipped `relates.nl` (`readout: decision`) |
| | `semanticRank(items, goal, { debiased?, concurrency? })` | concurrent pairwise schedule, symmetric debiasing `(a, b)` & `(b, a)`, exact Copeland/expected-win sort | shipped `beats.nl` (`readout: decision`) |
| Convergence | `semanticRepair(value, { check, fix, attempts })` | exact `check`, repeat detection, `iterateOn(...).withLimit`, closest-value fallback | caller's `fix` |
| | `fixedPoint(step, initial, { judge?, settled?, maxSteps?, carry?, view?, read? })` | revision loop through `iterateOn`, exact identity short-circuit, optional soft-state carry + final `read` | caller's `step`, shipped `settled.nl` (`readout: decision`) |
| | `semanticMerge(base, ours, theirs, options)` | three-way record walk (including add-add and edit-delete keys), kind + `check` verification, explicit `unresolved[]` | shipped `intentOf.nl`, `compatible.nl` (`readout: decision`), `combine.nl` |
| Policy | `escalate(fn, { to \| fallback, floor?, risk?, policy?, shouldEscalate?, history?, cost? })` | `pluggable({ crisp, nl })` policy (default `crisp`: asymmetric per-label risk floors before confidence floor; `shadow` serves crisp and emits `pluggable_shadow`), `decision_escalate` trace event, `.on({ model })` | caller's `readout: decision` function + shipped `shouldEscalate.nl` (`readout: decision`) when `policy` is `'nl'` or `'shadow'` |
| | `withContext(fn, entries)` | `fn.in(Context.ofCallable(fn).with(entries))` keeping companion helpers | caller's `fn` |
| | `compileRule(rule, { examples?, check?, attempts? })` | concurrent example validation + `semanticRepair` loop | shipped `compileRule.nl` returning `(item) => Promise<O>` |
| Workspace | `proposeAndVerify(folder, reducer, { verify, attempts, args })` | `folder.propose`, exact `verify`, `semanticRepair` over draft continuation, `folder.accept` | caller's reducer |
| | `branchAndSelect(folder, branches, { verify?, goal?, select?, debiased? })` | concurrent `folder.propose`, exact `verify`, `semanticRank` over passing branches, `folder.accept` | caller's branches + shipped `beats.nl` |
| Instructions | `instructionsOf(source)`, `stepsOf(body)` | front matter, declaration (without `readout:`), numbered items else paragraphs, fenced blocks kept | none |
| | `stepKind(step, signature, file)` | — | shipped `stepKind.nl` (`readout: decision`): `judgment` / `mechanism` / `guard-for-type` / `redundant`; examples in `std/examples.json`; read by the idiom reviewer and the program migrator |

## Traces

Each combinator run creates its own span `std:<name>` under the call that ran it, and runs its stages and judgments in
a child frame (`parentCallId === span.callId`). Every judgment call therefore nests directly beneath the combinator in
`natlang traces`, and concurrent combinators under `Promise.all` never mix their stage call IDs.
