# review: a function with helpers

The second program to read after `examples/hello`. `review.nl` assesses a list of observations against a criterion.
It does not do everything itself: it calls two helpers from its **callable folder**, one natural-language and one
TypeScript. This folder is a copy of the natlang-authoring skill's `assets/review/` example (a test keeps them equal).

```
review.nl            (observations, criterion) => Report: the function natlang.json runs
review/assess.nl     (observation, criterion) => Assessment: judges one observation
review/summarize.ts  (assessments) => Report: counts the verdicts exactly
types.ts             Assessment and Report, shared by all three
```

## The callable folder

A function may call exactly the items in its companion folder: `review/` beside `review.nl`. So the instructions of
review.nl can say "assess every observation ... with assess" and "return summarize of the assessments": `assess` and
`summarize` are functions in its scope, named after their files. Nothing outside `review/` is callable from review.nl,
and `assess` cannot call `summarize` (items of one folder do not see each other; nest a folder to give a helper its own
helpers).

## The TypeScript helper

Counting is exact work, so it is code, not instructions: `review/summarize.ts` default-exports an ordinary function. It
uses the types from types.ts without importing them, like the `.nl` files do. The model calls it from review.nl and
gets an exact `Report` back. `package.json` (`"type": "module"`) makes the TypeScript files ES modules.

## Check, call, test

```sh
cd examples/review
natlang check                       # types, signatures and call sites; no model
natlang run                         # review.nl on the sample in natlang.json
natlang call review.nl '["The backup took 42 minutes.", "The backup took 3 hours."]' "Backups finish within one hour."
natlang call review/assess.nl "The backup took 42 minutes." "Backups finish within one hour." --json
natlang test                        # the saved examples in examples.json
```

`natlang call` fills parameters in order: text as written, other types (the list of observations) as JSON. A helper in
the folder can be called on its own, as `review/assess.nl` above. Every call prints its call ID on stderr;
`natlang traces show CALL_ID` shows each step, including the calls to `assess` and `summarize`.

`natlang test` compares results exactly. An assessment carries a free-text `reason`, so examples.json saves only the
case whose result is exact (no observations gives zero counts); judge the assessments themselves with `natlang call`
and the trace, and save an exact case when you have one (`natlang traces fix CALL_ID --expected JSON`).
