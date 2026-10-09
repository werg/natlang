# Fused pipelines: settings, certificate, traces

Fusion keeps the value between two natural-language functions as a Neuralese block instead of text, when nothing else
reads it. It is a compilation decision: declared types stay `T`. Design and gate: `plans/FUSED_PIPELINES.md`.

## Settings (`natlang.json`)

```json
"fusion": { "mode": "off", "planner": "crisp", "certificate": "fusion-certificate.json", "weights": "<sha256 of served weights>" }
```

- `mode`: `off` (default; nothing changes, nothing is traced), `shadow` (text is served; the fused producer runs beside
  it and `fusion_shadow` records agreement), `on` (fused where certified, text otherwise).
- `planner`: `crisp` (default, exact rules, no model), `nl` (the `applications/fusion-planner` program decides, a crisp
  verifier checks), `shadow` (both, crisp is served). `NATLANG_FUSION_PLANNER` points at another planner program.
- `natlang run` plans at launch and installs the runtime option. Embedded runtimes: `createNatlangRuntime({ fusion: {
  mode, edges, certificate, weights } })` with `edges` from `fusedEdges(facts, plan)` (the runtime package's `fusion` subpath:
  `fusionFacts`, `crispPlan`, `planFusion`, `verifyPlan`).

## When `on` engages

Only when the runtime has a Neuralese store and port, the driver declares `neuralese: true`, and the certificate
(`natlang.fusion-certificate/1`, issued by the training pipeline after runtime qualification and the self-feedback gates
of the exact weights) matches model id, `weights`, and the port's dialect. Otherwise every planned edge stays text and
the orchestrator's trace gets `fusion_fallback` with the reason. A text emulation stands in only where the host sets
`fusion.emulation = { dialect }` and the port writes that dialect; events then carry `emulation: true`. Never ship that.

## What you see

- `natlang check [--fusion] [--json]` lists the candidate hand-offs and the plan. Edges kept as text give the reason
  (another reader, different models, a finite value, a chain the prose does not make explicit).
- Trace events on the orchestrator's call: `fusion_edge` (`fused`, `consumed`, `fallback`), `fusion_fallback`,
  `fusion_shadow`; the planner comparison is `pluggable_shadow` (name `fusion-plan`).
- A fused block is accepted where its declared type `T` is expected, in that task only. Code that reads it as `T`
  (field access, comparison, `String(v)`) is a reader the plan should have excluded; if you find one, the facts missed
  it: keep the edge as text and report the case.
- `scripts/fusion-bench.mjs` measures tokens, latency and agreement, and applies the gate before `on`.
