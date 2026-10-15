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

Native RL production independent follow-up:

- The shared-parent/two-learner merge and semantic refresh are now executable:
  updated native participants enter the existing shared VAE/joint/distill
  trainer, reset changed-identity exposure and export a newly bound adapter.
  Idempotent publication retains sibling updates and prevents duplicate update
  accounting. This resolves the earlier *implementation* gap in adaptive adapter
  lifecycle; actual provider execution and task-quality qualification remain open.
- Found and repaired actor admission that compared only the shared checkpoint
  SHA. Collection now attests the exact named participant, native identity and
  semantic adapter SHA both before and after sampling, and seals these expected
  bindings with the collection. A real persisted shared-model/VAE loader fixture
  verifies its receipt and rejects the wrong participant from the same file.
- Found and repaired readiness abort-on-child-failure. Semantic and RL failures,
  including missing executables, retain actual completion state and retry after
  a durable, adjustable poll delay (at most 60 seconds). Successful semantic work
  is not repeated merely because the later RL process failed.
- Found and repaired controller-restart duplicate work: ownership is written
  before spawn, recovered by unique launch token, argv/cwd/output and Linux
  boot/start identities. A surviving original process is monitored; orphaned
  descendants are checkpointed/drained before any new launch. Signals use pidfds.
  Exit status unavailable after reparenting is not guessed successful.

The focused actual-model/control regressions do not qualify a cloud worker or
learned task quality. The independent overall cutover review remains OPEN until
provider execution, both complete production recipes and runtime/quality receipts
are actually demonstrated.

Validation: `native-review-rl-actor-final-20261015` passed 39 focused
production/readiness/actual actor tests. After descendant-drain and signal
restoration changes, `native-review-rl-treecheck-20261015` passed 14 focused
review/readiness tests, including a real surviving subprocess tree. Quick checks
passed under `native-review-rl-quick-20261015`. The supervisor's single standing
full check remains the integrated full-suite authority; this review did not
launch a competing full suite.

Communication SFT/adaptive append review (2026-10-15):

- Independently loaded the admitted corrected first REST27 pair shard using its
  actual qualified model IDs and expanded tokenizer fingerprints. Both model
  views have 37 genuine nonempty graphs and 41 consumed occurrences; LFM imports
  1,229 payload tokens and Mellum 1,273. Every import has adjacent consuming
  boundaries with no plaintext query span inside them. This is partial REST27
  coverage, not a claim that all remaining source trajectories are converted.
- The shared trainer's real two-port fixture verifies one flat forward, exact
  cached consumer state/loss and consumer-to-producer adapter gradients.
  Empty authenticated imports remain structural rows with zero communication
  credit. Readiness requires complete positive genuine communication exposure
  matching its bound completed report before shared-space adaptation.
- Source manifests preserve exact physical prior prefixes and immutable row
  bindings. Real interrupted continuation/repacking retains optimizer/RNG state,
  per-row credits and repeated predecessor adoption. Rotating held checks preserve
  scope/cursor while final completion requires full held coverage.
- Fixed role-ID tuple/list cache-key mismatch: actual JSON cache save followed
  by a fresh ordinary/paired source now hits without parsing. Legacy cache reuse
  remains bound to the exact old trainer and unchanged parser/compiler/source.
- Added explicit nonnegative source training_priority. Communication always
  comes first; stronger-teacher ordinary sources can precede the older ordinary
  backlog without changing row identity, quota, weight, epochs or prior credits.
  The manifest SHA binds the priority policy; source index stays stable.

Validation: ledger native-review-priority-20261015 passed 23 focused trainer
checks, including real resume/priority order and both model ports. The subsequent
native-review-priority-cache-20261015 passed all 3 new cache/priority checks.
Native-review-admitted37c-20261015 independently validated all 37 corrected pair
rows for each actual participant. The first probe lacked the zstd container
binary; the second used friendly names instead of qualified model IDs; both
were corrected explicitly before the successful third probe.

A further adaptive restart gap was found while reviewing the priority change:
changing only the source manifest priority rejected the original already-adopted
predecessor because the new manifest SHA differed. Adoption now accepts its
previously verified, durable original predecessor digest across later manifests;
all old row bindings and the physical prefix are still checked before training.
The real resume regression changes priority between stops using the same original
predecessor and verifies exact remaining updates without replay.

Final `native-review-priority-adoption-20261015` passed 4 focused checks after
the adoption repair (real priority-tweaked resume plus both cache reloads and
priority identity checks). `native-review-priority-publishquick-20261015` passed
quick checks. No competing broad check or cloud process was started.

Advanced native learning-service follow-up:

- Repaired interface-only optimization: named backbone and interface parameters
  now use the existing composite storage/loader, shared standard SGD/Adam factory
  and training gradient/commit helpers. Publication clones selected/tied tensors
  and the module tree, preserves untouched storage, verifies the unchanged exact
  parent and never mutates the live actor.
- Repaired dropped optimizer history when requests alternate target sets:
  per-parameter Adam moments/steps and SGD momentum persist across updates and
  serving reloads. Shared RNG capture plus immutable training-state sidecars
  preserve continuation; the immutable parent retains its original trainer state.
- Repaired unbounded directory/recursion growth: updates share one directory and
  restore ancestor patches iteratively. Interface-only historical manifests
  remain loadable. Full named patch inventories and actual restored model
  configuration are verified.
- Native gradient requests previously ignored requested order2/derived and
  adapter fields despite absent support. These now fail explicitly before
  computation. Honest capability denial does not complete higher-order,
  adapter-value application or guided-generation implementation; those remain
  assigned open work. A changed native checkpoint needs semantic adapter refresh.
- TS clients compare exact identity fields independently of JSON property order,
  restore archived gradient/optimizer artifacts, and validate gradient, update,
  optimizer and returned-checkpoint bindings. Their HTTP/archive fixture covers
  restarted serving storage and rejects an inconsistent deployment identity.

Ledger native-review-fullupdatesc-20261015 passed 23 actual model/service checks.
Native-review-fullupdates-final-20261015 passed 12 final actual LFM/Mellum
SGD/Adam, checkpoint and unsupported-operation checks. Native-review-fullupdates-
node-final-20261015 built the TS host and passed 3 client/codec/persistent-store
checks. Initial new fixtures failed on absent/wrong chat templates and an
ordinary-only graph's legitimately zero producer-adapter gradient; they were
corrected to actual family templates and producing spans, without changing
numerical criteria. These are mechanism controls, not learned-quality evidence.
Both skills contain recovery guidance; live model-facing measurement is recorded
as debt. The cloud agent was told to include the new training_state sidecar in
provider dependency closure. Overall migration/cutover review remains OPEN.

Final chain-memory check native-review-fullupdates-chain-20261015 passed the
actual three-step LFM Adam reload: only the original full trainer parent and
latest optimizer state stay resident, rather than retaining every ancestor's
optimizer tensors. Quick checks passed under native-review-fullupdates-quickb-
20261015; the first larger quick reservation was refused before execution, then
rerun with the measured sub-1-GiB quick-check footprint and a 2-GiB claim.

Portable learned-actor follow-up:

- Native composite bundles preserve original manifest bytes and exact artifact
  IDs while carrying every parent, parameter, optimizer and RNG dependency.
  The shared ArtifactResolver now supports explicit digest-named content roots;
  native and semantic loaders and RL actor attestation use the same resolver.
  Corrupt supplied objects cannot fall back silently to an old source path.
- Optional semantic exports include the exact adapter receipt and all bound
  participant tokenizer files, with a complete verified inventory. Actual
  relocated shared actor writes retain the same compact message ID.
- Actual LFM/Mellum SGD/Adam tests delete original checkpoint/sidecar paths, move
  the HF source directory, reload identical identities/weights from the bundle
  and perform the next optimizer step. Inference request generator state yields
  the same next sample after reload. Deterministic service updates preserve
  training RNG; chat uses per-request RNG and does not advance that stream.
- ArtifactResolver's previous per-instance digest cache could hide later file
  mutation; it now uses the shared stat-bound hash cache. Tests reject altered
  and escaping content objects as well as corrupted RNG dependencies.

Validation: native-review-portable-updatesb-20261015 passed 16 focused checks;
with semantic export and deletion of original paths, native-review-portable-
semantic-20261015 passed 30. Final shared-resolver/relocated RL actor attestation
checks passed 3 under native-review-portable-final-20261015, and quick checks
passed under native-review-portable-quick-20261015. The first probe supplied an
incorrect test-file path and ran no tests; its corrected successor provided the
evidence above. Portability is actual checkpoint application, not a claim that
legacy direct adapter values, higher-order or guided native generation are done.

Native guidance repair (independent review, 2026-10-15 evening): actual native
serving reuses Guide's call/line checks. One prompt KV/hybrid anchor plus retained
actual own inputs/roles replaces per-token full cache snapshots. Rejection prunes
sample/write/ref effects, bans the rejected own input, and replays the retained
prefix with exact producer boundaries; request RNG continues. Streaming buffers
until commitment. Serving and graph policy objectives share one full-vocabulary
masked normalization helper; `/2` receipts carry sparse banned masks and keep
forced inputs separate. Both actual LFM conv/attention and Mellum ports pass
bitwise cached-reference restoration with imported state and retained/rejected
producer writes, stochastic continuation and accepted-action replay gradients.
Final HTTP/capture integration and consumer-only producer gradients passed in
ledger native-guidance-final-20261015 (16 tests); quick also passed.
Actual guidance-policy/forced-conflict regression is additionally tested. These
are mechanism checks, not learned task-quality or production-cutover completion.
Second-order and direct adapter-value operations remain open; native full-model
updates and portable bundles are independently checked mechanisms.
