# Delivery and integration scenarios

Make the application concrete and executable. Explain which judgments natlang makes and show an interaction in which it chooses operations from observed results. Include setup and run commands, model requirements, trace access, and unresolved gates.

## Verification ladder

1. `natlang check` passes: types, `nl` signatures, callable scoping, loop and recursion policy.
2. A scripted model driver exercises the real runtime: several calls, a rejected action and its repair, partial results, services, and a completed return. Label it wiring evidence.
3. Native operations and their failure contracts are tested on their own, including operation identity and unknown outcomes.
4. Interfaces: rendering, event handlers, persistence and reload, duplicate events, view failure plus `refresh`, and cancellation. Test browser behavior in a real browser (`npm --prefix ts-host run test:browser`).
5. A real interpreter runs a scenario that needs semantic judgment. Record provenance, token and latency figures, and semantic checks; do not conflate this with a fixture smoke.

## Design reviews that expose weak integrations

- Ask a notebook to resolve dependencies and decide what to recompute. Does natlang choose and inspect, or does a host switch statement do everything?
- Deliver a second event while a reduction runs. Is ordering explicit, and does the interface stay responsive?
- Let a service call succeed and then make the result fail validation. Is the effect preserved rather than blindly replayed?
- Commit a reduction and then fail its view. Is state durable, and does presentation retry without a second reduction?
- Change a captured variable from another task during a call. Does the application handle the conflict?
- Run a callable folder with a `while` loop or a recursive helper. Does `natlang check` reject it with a useful location?
- Generate an unfamiliar interaction. Are its controls bound to real handlers and included in export and import?

## Deliverable evidence

Report implementation, checks, fixture tests, live-model results, semantic review, and deployment separately. For teacher data, preserve provenance and independent semantic judgment; structural checks alone do not admit a sample. A discovered infrastructure defect is useful output: describe the failing contract and fix it in the shared layer.

Anchors: `ts-host/test/{runtime,project,browser,terminal-application}.test.mjs`, `ts-host/scripts/{studio-smoke,playground-smoke,browser-pilot}.mjs`, and each application's tests under `ts-host/test/`.
