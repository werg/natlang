# Support-ticket triage

A small application: two typed decisions and an inline summary do the judgment, TypeScript and `@natlang/node/std` do the bookkeeping, escalation, and repair.

```sh
natlang run examples/triage                                   # the bundled tickets.json
natlang run examples/triage -- my-tickets.json                # your own JSON list of tickets
natlang call examples/triage/classify.nl "My printer is on fire" "billing: charges and invoices; technical: anything broken; spam: advertising"
natlang test examples/triage                                  # the saved examples in examples.json
```

## Files

- **types.ts**: the shared types. `Label` is `"billing" | "technical" | "spam"`: every classification is one of these
  three, whatever the rubric says. `Summary` is `Is<string, "...">`: a property the runtime checks on every summary
  the model writes. `Report` is what the application prints.
- **classify.nl**: `(ticket: Untrusted<string>, rubric) => Label` with `readout: decision`: the model reasons, then
  scores the three labels. The rubric explains what belongs in each label; changing it changes how tickets are sorted,
  not which labels exist. To add a label, change `Label` in types.ts and the rubric together.
- **is_urgent.nl**: `(ticket: Untrusted<string>) => boolean`, also a decision, with the urgency rule in its instructions.
- **triage.ts**: ordinary TypeScript that imports the two `.nl` files as functions. `escalate(classify, { to: 'teacher', floor: 0.8 })`
  from `@natlang/node/std` scores locally and sends only tickets below 0.8 confidence to the stronger model (`teacher` is
  the profile your model profile names as `escalate`; without one, the local answer stands with `escalated: false` and
  `reason: 'unavailable:...'`). It checks urgency for
  non-spam tickets, counts labels in code, and shortens the refined inline `nl` summary with `semanticRepair`.
- **main.ts**: the `triage` target in natlang.json. Wraps incoming ticket strings with `untrusted(...)` and calls `triage`.
- **natlang.json**: the package: its files, the `triage` target and its arguments, the model, and warm-up calls.
- **examples.json**: saved inputs with expected results for `natlang test`.
- **bench/**: the benchmark pack (plans/semantic-fp/06 §4): `fixtures.json` (148 one-ticket units with gold label and
  urgency, train/held split by split group), `verify.py`, `tune_examples.py` and `bench.json` (the frontier arm's prose
  prompt). `scripts/semantic_fp/bench.py` runs the four arms on the held split; results are in `bench/results/`.

## Why the summary uses `semanticRepair`

The summary is shortened with `semanticRepair(draft, { check, fix, attempts: 3 })` from `@natlang/node/std`: an exact
word-count check drives up to three rewrites through `iterateOn(...).withLimit({ maxSteps: 3 })`, and if the bound is
reached it returns the closest honest summary instead of failing the report.
