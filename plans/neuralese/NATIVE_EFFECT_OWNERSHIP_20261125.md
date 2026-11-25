# Native descendant effect ownership (2026-11-25)

Separate from the valid Source35 cross-task graph: a TEST-only real compiled TypeScript entry calls an NL child, whose local scripted executor invokes a native service. Before this fix one physical invocation produces two effect receipts: child `agent` at graph_order 0 and ancestor `crisp` at graph_order 1. No provider, remote inference or teacher registry is used. Original probe and ledger: `/home/werg/vibe-cross-task-custody-ops/check-effect-ownership.mjs`, `vibe-native-effect-owner-20261125a`.

## Inventory before edits

- `ts-host/src/native/effects.ts`: shared service recorder, once cache, selected native-execution surface. Previously proxies hid their underlying authority, so a descendant wrapped an ancestor observer.
- `ts-host/src/runtime/native-entry.ts:50`: authored TS entry inserts recorded exact lexical services into its frame. Fixed through the shared recorder; previous bc8 grant subset correction retained.
- `ts-host/src/runtime/kernel.ts`: agent path wraps `checkedServices` (around line 785), crisp path wraps exact frame services (around line 1073). Agent now recovers the exact lexical port snapshot before applying *current* result checks and recording.
- `ts-host/src/native/refinement.ts:384`: result-checking proxy. Discovered separately that proxy identity incorrectly partitions ONCE_EFFECTS caches. Shared native invocation now caches on the physical service object, while every current result check still executes.
- `ts-host/src/native/runtime.ts:1137`: standalone execution records once; already-recorded kernel services retain their current observer. No extra layer needed.
- `ts-host/src/runtime/node.ts:133`: selected service surfaces adapt these same proxies for native bindings; surface/receiver behavior preserved.
- `ts-host/src/runtime/modules.ts`: native service imports use the executing frame. No independent effect wrapper; no change needed.
- `ts-host/src/runtime/{callable,contexts,iterate}.ts`: callable/combinator paths ultimately invoke the common kernel, not an independent recorder.
- `ts-host/src/runtime/context.ts:5,185`: Node async context and browser synchronous/callback propagation carry the same service frame; `ts-host/src/core/workerd.ts` uses async-local frames. Shared fix covers their kernel execution. Node and Workerd public execution tested; browser UI not run.
- `ts-host/src/tune/world.ts:9`: its journal proxy deliberately is not a recorder. It remains a caller-supplied semantic wrapper and is not stripped.
- Model engines in `docs/ENGINE_PARITY.md` (managed llama-server, page wllama, Python native, SGLang) produce model decisions; host service observation belongs to this common TypeScript runtime, independent of model transport. No model server or learner representation changed; no engine inference was run.
- Python/Node custody readers consume recorded effects, not service proxies; their exact hash/child inventory checks are unchanged. Cross-task replay ordering refusal remains unchanged.

## Change and boundaries

Private WeakMaps retain the exact granted service names and their original port identities. Descendants replace recorder-owned instrumentation instead of invoking through an ancestor's observer/checker. A new child applies its own current result checks. No task-wide service lookup, extra names, caller authority, new causal parents, digest waivers or retrospective packet edits are introduced. Later additions to an original mutable service map do not broaden the retained map.

The ONCE cache retains physical object identity, failure retry and native receiver semantics. A cached native result still passes the current invocation's result checker. Caller-authored adapter functions are opaque capabilities; this fix does not introspect or serialize arbitrary JavaScript closures. Intermediate function snapshots remain honestly nonportable.

## Validation

Official Node build and 38 targeted tests passed in `vibe-native-effect-custody-20261125b`, including real TS→NL concurrent Node/Workerd effect ownership, narrow native grants, retained higher-order context, exact causal occurrence, private receiver, exact grant snapshots, current result checks and ONCE scope. The first test run correctly caught the ONCE/refinement interaction and was not counted as a pass. The focused four-test suite, including native failure retry and cached-result checker rejection, passed in `vibe-native-effect-negative-20261125a` (212.5 MiB). Another 28 existing frozen-service, service-boundary, refinement and graph/replay contracts passed in `vibe-native-effect-contracts-20261125a` (608 MiB). The official build +38 tests +quick gate passed together in the b unit (2 GiB peak/36.5s).

Unchecked: real browser UI, remote engine inference, shared cross-task effect replay order, optimizer/held improvement. This independent correctness fix does not gate production collection or rewrite original observations.
