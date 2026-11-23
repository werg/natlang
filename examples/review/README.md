# review: a function with helpers

The second program to read after `examples/hello`. `review.nl` reviews a list of observations against a criterion.
It delegates exact fan-out and counting to its TypeScript helper (`review/summarize.ts`), delegates each per-item
judgment to `review/assess.nl` and its scored decision leaf `review/assess/verdict.nl`, and owns the top-level
semantic synthesis (`conclusion`). This folder mirrors `skills/natlang-authoring/assets/review/` (a test keeps them equal).

```
review.nl                 (observations, criterion) => Report: synthesizes the overall conclusion from the exact tally
review/summarize.ts       (observations, criterion) => Tally: maps assess concurrently in TypeScript and counts verdicts
review/assess.nl          (observation, criterion) => Assessment: calls verdict and writes a grounded reason
review/assess/verdict.nl  (observation, criterion) => "supported" | "contradicted" | "uncertain" (readout: decision)
types.ts                  Assessment, Tally, Conclusion (Is<string, ...>) and Report
```

## The callable folder

A function may call exactly the items in its companion folder: `review/` beside `review.nl`. `review/summarize.ts`
imports its sibling `./assess.nl` to fan out in parallel with `Promise.all` and tally counts in TypeScript, so the
language model is never asked to orchestrate a deterministic loop. `review.nl` calls `summarize` and spends its model
call on genuine synthesis: judging whether the criterion holds overall.

## The decision

`verdict.nl` returns one of three labels and declares `readout: decision`, so the model reasons and then scores the
three values with calibrated probabilities (and `examples.json` tests it exactly). `assess` keeps the free-text reason.

## Check, call, test

```sh
cd examples/review
natlang check                       # types, signatures and call sites; no model
natlang run                         # review.nl on the sample in natlang.json
natlang call review.nl '["The backup took 42 minutes.", "The backup took 3 hours."]' "Backups finish within one hour."
natlang call review/assess.nl "The backup took 42 minutes." "Backups finish within one hour." --json
natlang call review/assess/verdict.nl "The backup took 3 hours." "Backups finish within one hour."
natlang test                        # the saved examples in examples.json
```
