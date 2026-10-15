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

Native client completion follow-up:

- Forced typed strings and contextual bodies now call real native producer
  routes, verify exact text/token/role/source identities, archive output/imports
  and preserve captures through `.nz` provenance. Empty strings use zero rows
  with the complete model/layer layout; imported zero spans are cache no-ops.
- Native views run the existing natural-language single/part/combine stages
  through actual native template writes. The endpoint retains every source
  turn; the client archives every intermediate and records shared graph edges.
  Context-fit grouping reduces large sets of part states structurally.
- Focused tiny LFM/Mellum forwards and deterministic sample controls test the
  mechanism. They do not qualify a trained model's task performance.

Advanced native learner-service remains an explicitly assigned repair, owned by
root for scheduling and shared native-objective integration. These are live
consumers, not deprecated features to silently strip:

| Consumer | Required service | Required native validation |
|---|---|---|
| `neuralese/learning.ts` `evaluate`, `grad`, `valueAndGrad` | `POST /v1/neuralese/grad`, first and second derivatives | Whole native invocation graph objective, native parameter/checkpoint identity, exact gradient agreement; reuse `test/neuralese-learning.test.mjs` cases on both models. |
| `neuralese/learning.ts` SGD/Adam optimizer methods | `POST /v1/neuralese/optim` | Real native parameter updates, optimizer resume and changed forward behavior; no projection/head alias. |
| `neuralese/learning.ts` adapter creation and `withAdapters` | `POST /v1/neuralese/adapters` plus request adapter application | Explicit native adapter representation, correct checkpoint identity and functional/gradient validation. |
| `model/neuralese-server.ts` decision readout and batch scorer | `POST /v1/neuralese/decide`, `POST /v1/natlang/score` | Exact cached native candidate scoring with typed imports, per-item and batched agreement; exercise `scripts/skills/decision-lib.mjs` and graded decision tests. |
| `neuralese/learning.ts` distribution/delta/memetic methods | native multi-tensor delta/parameter operations | Preserve model/layer identities and tensor semantics through diff/apply/compose/interference/learnMerge/save; native KV is not a flat legacy vector. |
| Native guided requests through the ordinary agent | `guidance` chat operation | Native cache/state rollback and sampled trajectory correctness, not merely syntax checking. |

Structured unsupported capability errors keep these paths honest; they are not
completion of the requested architecture replacement. Production cutover is
OPEN until the integrated functionality, executable cross-model stages, data
coverage and real model training/task checks are independently reviewed.

Validation for native client/zero/view follow-up: ledger `native-review-complete-final-e2e-20261015` ran 59 passing CPU tests and 8 GPU skips; `native-review-complete-final-node-20261015` ran 49 passing Node tests. Final quick ratchets/typecheck passed. Actual full production-backbone GPU qualification remains separate.

Final narrow follow-up: `native-review-publish-e2e-20261015` passed6 actual two-architecture tests after contextual length checks; `native-review-empty-dtypes-20261015` passed4 FP32/BF16 LFM/Mellum zero-state tests. Empty attention/convolution payloads retain the normal capture FP32 dtype even when embeddings are BF16.

Native body BPE correction: marker-free ordinary-text concat equality was an
obsolete identity requirement. Actual producer markers delimit prefix, payload
and suffix token runs, so valid contextual native bodies no longer fail when an
ordinary string would merge across those boundaries. Actual payload/source
captures remain verified. Ledger `native-review-native-bpe-body-20261015` passed
four real LFM/Mellum handoff and BPE forced/body/view cases.

Compact semantic wire review (2026-10-15):

- Actual source token IDs cannot be replaced by tokenizing their decoded text.
  The review found that serving rejected valid noncanonical/byte-fallback
  sequences. The shared offset helper now preserves actual IDs; a real HF
  ByteLevel regression covers split `hello` and partial UTF-8 `€` tokens.
- Compact producer replay must archive its raw source capture; decoded consumer
  observations must also be archived. Both use the existing exact native-store
  lifecycle. Two actual LFM/Mellum incremental producer fixtures pass durable
  compact+raw bundle, policy-logp and gradient replay checks.
- View writers preserve every compact message, raw producer capture and decoded
  observation through all part/combine turns. Contextual body provenance must
  retain both original imports and actual observed artifacts, including the
  exact adapter/source/target binding; `.nz` follows both dependencies.
- The shared dialect is `semantic-state/<adapter SHA>`. Native and semantic
  DefaultDialect values require explicitly advertised reencoding/adaptation;
  named dialects stay strict. Native libraries retain their actual semantic
  dialect and model-bound zero identity instead of creating old vector empties.

Compact follow-up validation: ledger `native-review-compact-archive-e2e-20261015`
passed 9 tests, including actual LFM/Mellum compact forced text/context/empty
capture, original+decoded `.nz` dependencies, compact chunk/combine views,
ordinary server receipt-only archival, durable producer RL replay and real HF
ByteLevel offsets. `native-review-compact-archive-node-20261015` passed 8 typed,
archive and semantic-library tests; quick checks passed. VAE weights in these
fixtures are controls, not evidence of learned task quality.

Adaptive adapter lifecycle remains OPEN: a later SFT/RL checkpoint update changes
participant identity. Explicit shared-adapter continuation/retraining and bound
export must accompany that change; a stale adapter must not be silently relabeled
or leave the next actor startup permanently blocked.
