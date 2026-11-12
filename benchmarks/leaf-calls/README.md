# Leaf calls: interpreter surface vs direct answers

A leaf call has nothing to run code on: no callable functions or services, no files, folders or Neuralese, and arguments
that are plain data shown whole. Under the full interpreter surface such a call still receives the whole tool manual
(about 3.7k tokens of system prompt plus 1.1k of tool schemas for a one-label classifier) and answers through an eval
round trip. `leafCalls: 'direct'` (runtime model setting, `native/agent.ts` `leafOpening`) gives it a short system
prompt (`LEAF_SYSTEM_PROMPT`), an opening that shows its arguments, and only `return_result`. `decisionReadout:
'finite-returns'` answers finite-typed calls by scoring their values (`native/decision.ts`).

`suite.json` holds 24 leaf cases with exact answers: finite labels, booleans, language codes, extraction into strings,
records and lists, and two arithmetic cases (the risk of answering without eval).

```sh
# one variant per process, on the configured runtime's selected profile (the managed local model by default)
node benchmarks/leaf-calls/measure.mjs --variant tools    --samples 2 --out results/tools.json
node benchmarks/leaf-calls/measure.mjs --variant direct   --samples 2 --out results/direct.json
node benchmarks/leaf-calls/measure.mjs --variant decision --samples 2 --out results/decision.json
node benchmarks/leaf-calls/measure.mjs --compare results/tools.json results/direct.json
```

`--compare` pairs results by (case, sample) and reports McNemar's exact test with each variant's accuracy, latency,
turns and tokens. Results and the decision they support are recorded in `plans/MODEL_FACING_CHANGES.md`.
