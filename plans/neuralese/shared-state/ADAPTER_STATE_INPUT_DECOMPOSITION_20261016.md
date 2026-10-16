# Explicit native state exchange between exact-base adapter actors

Natural-language contract: a caller may explicitly request sharing native state
between different adapter actors of the exact same base model. The consumer is
told that imported KV/hybrid state comes from another adapter actor; no semantic
interoperability between different bases or architectures is implied. Default
transport keeps existing strict identity/learned-adaptation/text-reencoding rules.

Caller declaration is an input-ID keyed `native_state_modes` map, carried by
chat as `x_natlang_native_state_modes`, and by native learning/scoring/forced
context operations in their typed request. The only new explicit mode is
`same-base-adapter-state`. Unknown modes/absent input IDs fail. A requested mode
never silently turns into text reencoding.

Exact mechanisms and validation:

1. Verify the immutable raw native-state artifact and its source effective
   identity. Resolve the actual source coefficient/basis closure if adapted.
   Both source and current target must share exactly the same checkpoint base,
   tokenizer, embedding identity and architecture. Validate the actual layer,
   head, convolution-state and position profile against the consumer.
2. Splice the source's actual raw state at consumer-local positions using the
   existing positional/hybrid-state mechanism, preserving source metadata.
   Record source and target actor/base bindings, source content ID, actual target
   start and length in a versioned consumption receipt. This is a distinct mode
   from exact-same-actor state and learned semantic adaptation.
3. Extend graph imports with per-input source identities; keep target actor
   verification explicit at the service/replay boundary. One forward uses the
   exact original state tensors; no width inference or raw-state relabelling.
4. Capture/replay retains actual per-input consumption receipts and source
   identities. Native-policy objectives evaluate current same-basis coefficients
   against immutable behavior logps and use this mode only when explicitly
   declared, with source/target contracts revalidated under the current scope.
5. Preserve both actor identities through typed clients, durable archives,
   checkpoint/runtime receipts and `.nz` dependency closures. Named dialect
   checking remains strict; this is not a new shared cross-base dialect.
6. Validate actual two-port cached/flat forward and coefficient/state gradients,
   current/behavior multiupdate runtime parity, source/target/tensor/profile
   mismatches, zero-state and convolution continuation, archive restart and
   explicit mode failure. Then extend heterogeneous per-invocation graph
   projections/FFNs without recursive graph replay, preserving producer gradients.

Policy is this declared service contract and natural-language recovery advice.
Crisp code authenticates identities, transports bytes, constructs exact masks,
applies proven splices and verifies results. Capabilities remain unadvertised
until integrated inference/training paths and independent review are complete.

Implementation status: explicit mechanism only; capability remains unadvertised.
Four both-port source/receipt/inventory cases passed (11.84s), and six CPU
retained-behavior continuation/durability cases passed (38.74s). Actual CUDA
current Engine prefill/import/incremental continuation versus second-update
flat objective passed LFM's strict coefficient test. Mellum **FAILED** that same
gate: max absolute coefficient difference 3.990344703197479e-5 and maximum
relative coordinate difference .0013894966104999185. No tolerance was changed.
Evidence: `/home/werg/data/dgxops/native-adapter-state-runtime-parity-20261016.log`.
This is an open adjoint qualification blocker, not a claim of runtime parity.
Run its exact strict qualification with `NATLANG_NATIVE_ADAPTER_TEST_DEVICE=cuda
NATLANG_REQUIRE_MELLUM_ADAPTER_STATE_PARITY=1`, pytest this file's adapted-policy
bundle/restart test selected with `-k explicit`. Default durability coverage does
not enable that known-red numerical certificate. Continue localization of actual
hidden/logits/logps and weight cotangents before adoption. Heterogeneous
per-invocation layer parameters remain open separately; no recursive replay is
being substituted. Local Torch2.11/SM121 checks do not qualify cloud H100/NGC.
