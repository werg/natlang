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
- `observed` (absent = off): `true` or `{ "minRuns": 20, "store": "<path>" }`. Readers the source does not prove are also read
  from the eval code orchestrating models ran, as the call store (default: the machine's) recorded it for the orchestrator
  exactly as it is now (same instructions, model-driven runs only). An edge fuses on that evidence only with at least
  `minRuns` supporting runs and none that read the value elsewhere. Plans carry `evidence` (runs, minimum, store revision).
- `natlang run` plans at launch and installs the runtime option. Embedded runtimes: `createNatlangRuntime({ fusion: {
  mode, edges, certificate, weights } })` with `edges` from `fusedEdges(facts, plan)` (the runtime package's `fusion` subpath:
  `fusionFacts`, `crispPlan`, `planFusion`, `verifyPlan`).

## Crisp TypeScript orchestrators

Hand-offs in your own TypeScript (`analyze(await parse(source))`, or `const tree = await parse(source)` used only as
`analyze(tree)`) are analyzed by syntax. The project build marks the calls of such hand-offs (`__natlang.fuseSite`); with
fusion off the mark only calls the function. A planned edge names the two call sites, and the runtime engages exactly
those calls. Anything else that touches the value (a read, a second consumer, an `export`) keeps the edge as text. The
declared types stay `T`: TypeScript code between the two calls would see a block, which is why nothing may sit between them.
Embedded hosts compile their TypeScript with the project build (`buildProject`) for the marks to exist.

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
