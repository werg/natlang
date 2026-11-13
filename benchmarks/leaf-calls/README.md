# Leaf calls: accuracy and generation length on the interpreter surface

A leaf call has nothing to run code on but plain data: no callable functions or services, no files, folders or
Neuralese. Every call gets the full interpreter surface; the model may answer directly (a reply without tools) or use
eval for exact work. `decisionReadout: 'finite-returns'` answers finite-typed calls by scoring their values
(`native/decision.ts`).

`suite.json` holds 24 leaf cases with exact answers: finite labels, booleans, language codes, extraction into strings,
records and lists, and two arithmetic cases.

```sh
# one variant per process, on the configured runtime's selected profile (the managed local model by default);
# --label names a run, e.g. one profile with a thinking budget (NATLANG_PROFILE)
node benchmarks/leaf-calls/measure.mjs --variant tools    --samples 2 --out results/tools.json
node benchmarks/leaf-calls/measure.mjs --variant decision --samples 2 --out results/decision.json
node benchmarks/leaf-calls/measure.mjs --compare results/tools.json results/decision.json
```

`--compare` pairs results by (case, sample) and reports McNemar's exact test with each variant's accuracy, latency,
turns and tokens. Results and the decision they support are recorded in `plans/MODEL_FACING_CHANGES.md`.
