# Causal child capture membership

The original 0066 `open-goal-program-run-dedupe-audio-recording` manifest (SHA256 `390f469a743ffe02144e6b873b8b9737df1ad6f620bf02e6f5e32bd06f576a6d`) contains 21 calls in one connected tree. Every recorded child inventory exactly matches the actual `parent_call_id` edges. Its application root explicitly records a fresh task's `check` as a child. Python and Node rejected this solely because they additionally required child and parent task IDs to be equal. Task identity separates execution authority and effect ordering; it does not prohibit an authenticated causal child.

## Shared mechanism inventory

- `scripts/neuralese_data/live_teacher_sources.py::verify_observed_child_inventory` checks each record's own task/custody identity, recomputed custody digest and exact child set. `load_episode` separately binds every captured call to its artifact SHA, independent call scope, owner/run, summary, root attestation and full ancestry. Remove only the cross-edge task-equality assumption.
- `ts-host/scripts/observed-workspace-evidence.mjs::verifyObservedCallTree` traverses the complete rooted graph, rejects duplicate/unreachable calls and checks custody with the shared digest implementation. The full disk verifier separately authenticates original bound artifacts and independently verified scope. Remove the same cross-edge task-equality assumption here.
- `ts-host/src/calls/graph-custody.ts::graphCustodyMatches` already correctly authenticates each call against its own task and exact child IDs, effects and parent. It does not require every child to share that task. Preserve it unchanged.
- The runtime's `ts-host/src/runtime/runtime.ts::registerGraphCall` forwards a cross-task child to its authenticated external trace parent. Kernel/native entry recording and lifetime checks retain fresh task services, model budget and cancellation boundaries. No producer or authority change is made by this reader fix.
- Host's exporter captures actual call IDs/parent edges and independently attested owner/run scope; Base invocation observations retain causal ancestry. The original packet already passed those producers and its exact graph matches its custody. They need no source change for this mismatch.
- `ts-host/src/calls/replay.ts::subtreeEffects` deliberately still refuses cross-task portable replay: it requires one task-wide effect-order namespace. The capture-membership repair must not invent a total order across independent task effect counters. Observed per-call learning and portable effect replay have different requirements.
- The TS observed-workspace row converter consumes the already authenticated observed graph; it adds no same-task condition. Python and Node offline readers are the affected acceptance boundaries.
- Managed llama-server, in-page wllama, Python native serving and SGLang are model request/score engines, not readers of Vibe workspace call artifacts (`docs/ENGINE_PARITY.md` engine inventory lines 9–12, score protocol lines 23–25). No model wire, model choice, retry, tokenizer, learner or serving change applies here. Shared runtime capture semantics remain the same across Node/Workerd/browser.

## Validation and boundaries

`vibe-causal-child-reader-20261125a`: 144 Python reader tests and six Node chronology tests passed, 6.5 seconds/424.1 MiB. New controls accept a truthful fresh-task causal child while rejecting mismatched own custody, missing/foreign-parent children and duplicate identities; portable replay still refuses independent effect namespaces.

The exact original manifest loaded with `task_completed:true`, 21 calls and 26 decisions. Canonical Node original-artifact verification passed, then the real materializer produced 26 eligible turn records from one row, zero rejected rows. Derived outputs are in an owned private operational directory; original capture/receipt/source bytes were not edited. No inference, network probe, reviewer call or teacher replay ran. This establishes this packet's reader correction, not all future graph/replay features or any provider-failure cause.

## Python digest parity follow-up

The inventory also found that Python checked the digest shape while the canonical Node reader recomputed its contents. Python now recomputes the same closed custody/effect-reference protocol using existing canonical JSON bytes (fixed ASCII field names, integer order/byte fields and preserved Unicode string values). A cross-language regression invokes the actual Node graphCustodyDigest for Unicode/lone-surrogate values and missing/empty effect fields; a corrupt digest is rejected. Existing synthetic custody fixtures now carry valid computed digests. No captured digest is regenerated or replaced.

`vibe-causal-child-digest-20261125a`: 145 Python tests plus six Node tests passed,5.7seconds/325.6MiB. The original21record manifest again passed Python and Node; the derived trajectory bytes were identical to the earlier materialized input. This follow-up does not add portable cross-task effect replay or change producer serialization.
