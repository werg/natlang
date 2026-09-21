---
name: natlang-integration
description: Embed natlang in Python, Node/TypeScript, and browser applications. Use when wiring model drivers, crisp engines and host objects, event streams, native frontend reducers and generated UIs, persistence, resumption, traces, or application evaluations.
---

# Integrate natlang applications

Build an application in which natlang can drive operations, inspect their results, and choose what happens next. Use the host for concrete execution, storage, presentation, and transport. Do not reduce natlang to a classifier attached to a host-authored workflow unless that is the requested product.

Start by locating the installed natlang version or source checkout and existing embedding. Reuse its shared runtime and model lifecycle. The supplied references travel with this skill; repository paths mentioned inside them are lookup hints relative to a natlang checkout, not paths relative to the installed skill.

## Choose the smallest adequate boundary

Read [hosts and model adapters](references/hosts.md) for Python/Node and driver contracts; [frontend applications](references/frontend.md) for browser model loading, event reducers, and generated interfaces; [effects and recovery](references/recovery.md) for native objects, streams, traces, concurrency, and durability. Read [delivery scenarios](references/delivery.md) before claiming an integration complete.

1. Describe who owns state, the semantic algorithm, native objects, and effects. Choose the runtime, evaluator, and model independently. An engine name selects an implementation available in that embedding; it does not install a new language backend.
2. Pass typed values or checked source definitions to the runtime. Expose host operations through crisp helpers or declared effect callbacks. Let natlang call them repeatedly and inspect observations. Host objects may hold databases, jobs, binary data, or a stronger model client without extending the language core.
3. Define event ordering and commit points. Streams/folds or a per-event reducer are the starting point; ambient interrupts and coroutines are not prerequisites. Keep portable state explicit across conversation rollover.
4. Implement model lifecycle, durable state, operation receipts, cancellation behavior, recovery, and the actual UI/CLI. Validate both success and partial failure. A correctly typed return does not certify the domain result.
5. Test the target engine and transport. Compare shared contracts across supported hosts when changing infrastructure. Make general fixes available to other callers; avoid an app-local workaround around your own runtime bug.

## Operational commitments

- Omit arbitrary run budgets unless the deployment requires them. Support productive long trajectories with state-based continuation and efficient data access. Conversation segmentation and host context allocation are separate controls.
- `BrowserNatlangApplication` defaults to local validation repair; low-level hosts and Python `ToolAgent` default to caller feedback at this revision. Choose policy explicitly for evaluations. Local feedback preserves type checking and successful earlier effects.
- Shared `host` access is trusted native execution. `fresh` globals do not make a host object sandboxed. Choose isolation by actual authority and threat model; sharing may be precisely what the application needs.
- External effects can outlive cancellation or result validation failure. Persist operation identity and observations; do not blindly retry an unknown result.
- Sampling seeds and versioned programs support reproducibility. They do not make arbitrary native state replayable or guarantee identical decisions on different inference backends.
- Verify live-model quality separately from fixture wiring and browser rendering. Name remaining empirical gates plainly.

Pair with `natlang-authoring` for substantial `.nl` algorithms when that skill is available. This integration skill is independently usable.
