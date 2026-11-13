# Support-ticket triage

A small application: two natural-language functions do the judgment, TypeScript does the bookkeeping.

```sh
natlang run examples/triage                                   # the bundled tickets.json
natlang run examples/triage -- my-tickets.json                # your own JSON list of tickets
natlang call examples/triage/classify.nl "My printer is on fire" "billing: charges and invoices; technical: anything broken; spam: advertising"
natlang test examples/triage                                  # the saved examples in examples.json
```

## Files

- **types.ts**: the shared types. `Label` is `"billing" | "technical" | "spam"`: every classification is one of these
  three, whatever the rubric says. `Report` is what the application prints.
- **classify.nl**: `(ticket, rubric) => Label`. The rubric explains what belongs in each label; changing it changes how
  tickets are sorted, not which labels exist. To add a label, change `Label` in types.ts and the rubric together.
- **is_urgent.nl**: `(ticket) => boolean`, with the urgency rule in its instructions.
- **triage.ts**: ordinary TypeScript that imports the two `.nl` files as functions. It classifies every ticket
  concurrently, checks urgency for the tickets that are not spam, counts labels, and writes a short summary of the
  urgent ones with an inline `nl` function.
- **main.ts**: the `triage` target in natlang.json. `natlang run` calls `main(context)` with a `TargetContext`:
  `context.arguments` holds the declared arguments (`file`, `rubric`, `tickets`) already read and type-checked, with the
  rubric's default applied; `context.package.root` is this folder (for the bundled tickets.json);
  `context.runtime.run(...)` runs natlang calls with the selected model; `context.io.output` is where the report goes.
- **natlang.json**: the package: its files, the `triage` target and its arguments, the model, and warm-up calls.
- **examples.json**: saved inputs with expected results for `natlang test`.

## Why the summary uses `withLimit`

The summary is shortened with `iterateOn(shorten, draft).until(text => words(text) <= 60)`: rewrite until it is short
enough. Nothing guarantees a model's rewrite gets shorter each time, so the loop needs a bound it can always reach.
`withLimit({ maxSteps: 3 })` stops after three rewrites with an `IterationLimitError`, and triage.ts then keeps the last
(shortest honest) summary instead of failing the whole report.
