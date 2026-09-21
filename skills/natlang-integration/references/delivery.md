# Delivery and integration scenarios

Make the user's application concrete and executable. Explain which responsibilities run in natlang and show an interaction in which it chooses operations based on observed results. Include actual setup/run commands, source/model requirements, trace access, and unresolved gates.

## Verification ladder

1. Load real source and bind realistic typed inputs on the intended host. Verify a missing helper/engine produces a useful failure.
2. Use a fixture model driver to exercise multiple calls, rejected-action feedback, partial results, and a completed return through the real runtime. Label this wiring evidence.
3. Test the native operation implementation and its failure contract separately, including operation identity and uncertain outcomes for external effects.
4. For frontends exercise rendering, event handlers, persistence/reload, duplicate events, view failure plus `refresh`, and cancellation. Use a real browser for browser-only behavior.
5. Run a selected real interpreter against a scenario requiring semantic judgment. Record actual provenance, token/latency metrics, and semantic checks. Do not conflate a fixture smoke with this gate.

Use relevant existing suites rather than blanket testing after every prose change. Shared runtime changes warrant regression and parity checks. Model evaluation should retain failure information and exact adapted requests even when the run throws. Persist incremental observations for long jobs so a machine restart does not erase the useful evidence.

## Design reviews that expose weak integrations

- Ask the notebook to resolve a dependency and decide what to recompute. Does natlang traverse and inspect, or does a host switch statement do everything?
- Ask a media program to inspect a failed transformation and revise the command. Can it read the real result/job, and is a large binary kept out of the prompt?
- Deliver a second event while a reducer runs. Is ordering explicit, and does the interface stay responsive?
- Let a host effect succeed then make its result fail validation. Does recovery preserve the unknown/successful effect rather than blindly replaying it?
- Complete a reducer then fail its view. Is the state durable, and can presentation retry without a duplicate reduction?
- Resume after a conversation rollover with a partially completed loop. Does it continue from the accumulator rather than restart?
- Generate an unfamiliar interaction. Are the visible controls bound to actual versioned handlers and included in export/import?
- Run the same scenario under two supported hosts. Do their value/error/effect contracts agree? If portability is intentionally limited, is the limitation explicit?

## Deliverable evidence

Report completed implementation, fixture tests, live model results, semantic review, and deployment separately. For teacher data, preserve provenance and independent semantic judgment; structural audit alone must not admit a sample. A discovered infrastructure defect is useful output: describe the failing contract, repair it at the appropriate shared layer, and verify other callers benefit.

Repository anchors: `ts-host/test/browser-application.test.mjs`, `ts-host/test/host.test.mjs`, `ts-host/test/native-parity.test.mjs`, `ts-host/test/research-runtime.test.mjs`, `tests/test_execution_boundary.py`, `tests/test_continuation.py`, and the relevant application's evaluation module. Current API declarations take priority when a historical example uses a different option name.
