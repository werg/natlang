# Independent native-state integration review

Reviewer: `native_state_independent_review`, separate from the graph implementer.
Status: OPEN. This review does not approve production cutover.

The graph executor performs a real layer-major forward over unique physical
query rows, with native raw keys and values, positional import masks and finite
convolution state. The normal path does not recursively replay producers.

Concrete gaps found and repaired or assigned:

- Logical producing spans did not transform their own query embeddings. Fixed
  in `47d8c56c`; mixed own/imported regressions compare exact graph outputs and
  adapter gradients to explicit own-token producing spans, plus cached
  execution, across LFM, Qwen and Mellum.
- Native IDs survived HTTP history but were rejected by typed Neuralese values,
  literals, archive restart and `.nz` imports. The repair extends the existing
  value concept, metadata/dialect checks, compiler literals, Memory/Node/OPFS
  archives, exact native codecs and compiled-import adoption.
- Native transport converted text parts into bare strings, causing the server
  to render entire content arrays as ordinary text. A successful reply imported
  no state. The repair preserves typed text/native parts; the actual two-model
  harness fixture asserts a real cache splice and catches this failure.
- Native agent decoding ignored direct native parts and used an unrecognized
  event kind. The operations agent owns the correction to the shared
  `block_write`/`block_read` events, source identities and actual producer edges.
- Full Mellum outputs and incremental state match bitwise, but a consumer
  adapter-gradient difference remains under investigation. Independent
  full-model backward hooks ruled out checkpointing as the sole cause and
  localized the first difference to layer 22 consumer-key adjoints. The core
  arithmetic agent is repairing the attention backward path; no tolerance was
  relaxed and no full-model gradient qualification is claimed here.
- Trainer eager corpus loading, unsafe multi-forward batching, absent epoch in
  checkpoints, frozen optimizer option guards and invalidated task-receipt
  hashes were reported to the trainer owner. Actual trainer update/resume
  evidence is still required beyond metadata/helper tests.
- Forced text writes, contextual body initialization and views need native
  client/server paths; work is assigned to this reviewer and operations agent.
  Capability errors alone do not complete these features.
- Native RL learner, complete data/recipe coverage, shared-space VAE/joint/
  distillation stages, two-model readiness triggering and adapter-bound learned
  inference remain root-owned integration work. Layout-only VAE code and stage
  names do not qualify them.

Focused evidence uses the DGX memory ledger. The typed runtime fixture executes
actual tiny LFM hybrid and Mellum HTTP model forwards, producer capture, typed
child results, persistent archive reopen, deliberate server eviction/restoration,
and a parent's exact native-state splice. Sampling/tool syntax is controlled:
this establishes runtime/state correctness, not learned task quality. Require
both the actual production backbones and the full integrated recipe before
cutover. New recovery messages retain live measurement debt.
