# Native checkpoint evaluation away from cloud training

Policy: evaluate the newest useful verified LFM export on the spare DGX GPU.
An evaluation already admitted or running finishes; newer exports coalesce while
it runs. Results describe their historical weights and never qualify different
current training weights. Both cloud trainers retain resource ownership.

1. Read the existing periodic exporter's verified receipt without modifying its
   state. Use its existing private owner-authenticated transfer, exact SHA/size
   check and immutable local artifact. No Mellum checkpoint download.
2. Seal the existing TaskEval asynchronous request and queue pointer. Keep the
   original full LFM artifact SHA; do not use legacy head or projection loaders,
   or prune into a different purported checkpoint identity.
3. Restore the exact shared native model with its declared arithmetic and sparse
   row dtypes, verify pinned base/model/tokenizer contracts, and run the existing
   held task collector. Bind runtime/protection and sampling identities in the
   durable request/result and existing task-score/log outputs.
4. Existing ledger dispatch owns one request's retry, execution and cleanup.
   Infra remains unscored; pending/superseded work is not model failure. Keep
   running artifact references until a report/drain; observe later exports after
   it finishes. Trainer final exact-weight qualification remains independent.

2026-10-16 correction: both students use this mechanism. Mellum uses an exact
verified weights-only serving view with its original full training-parent hash
recorded separately; full resumable exports remain unchanged. A derived view is
never a training continuation parent. New native requests and live native
evaluators require `neuralese_server=true`: the plain OpenAI collector has no
native store/value runtime and cannot transport emitted shared-state values.
Historical false-mode results retain their original protocol and are diagnostic,
not evidence of working native transport. Interrupt an affected diagnostic when
the mismatch is established and restart useful evaluation with explicit corrected
provenance. Existing request and sampler bindings must never be edited in place.

All steps above are crisp transfer/identity/durability/process plumbing. This
extends existing task_cadence and task_eval_dispatch, not another eval queue,
trainer, collector, provider account or task-selection policy.

2026-10-16 memory scheduling correction: new native evaluator unit names include
their declared backbone architecture, so the memory ledger learns separate job
families for LFM and Mellum. A Mellum peak must not inflate every LFM request's
reservation and preempt useful data work. Unit naming is crisp accounting, not
a different evaluation or selection policy. Existing dispatch records retain
their exact unit identity across dispatcher upgrades; never duplicate an active
or queued evaluator because its naming convention changed. Legacy request names
remain unchanged. Budget estimation and measured memory guards still apply.

2026-10-16 snapshot ownership repair: arbitrary local rolling checkpoints are
sealed through the existing verified independent-copy mechanism. Controlled
incoming downloads transfer ownership by rename (verified copy across devices);
no mutable hardlink aliases are created. Release unneeded loaded training
schedule metadata before runtime sealing. The LFM dispatcher exceeded its old
3 GiB cgroup during full-checkpoint ingestion; restart with an 8 GiB ledger claim
and preserve its existing export, queue, request and lease state. Focused async
evaluation tests: 20 passed, including mutation isolation and transfer ownership.

Assistant-provenance correction, 2026-10-16: `neuralese_server=true` alone is
insufficient. Frozen runtime `43e248a8...` flattened genuine assistant wire
parts into plain text before the typed return boundary. Two actual LFM7500
captures returned strings matching real same-turn native block writes. This is
a protocol defect; do not infer model inability or hypothetical repaired success.
Canonical fixes `bdcfb630e`, `2027cfdb2`, and `705771bd6` preserve actual assistant
parts, apply typed `return_result`, and prevent incidental writes/compaction
from completing a call. Independent actual-HTTP tests passed four cases and
compaction tests passed two. Future actor/evaluator runtime closures must include
these mechanisms; an old frozen runtime does not inherit source-tree fixes.

The affected7500 diagnostic was stopped with14 completed captures and2 unknown
cases. Its exact captures, request and weights remain assigned to runtime/cache
investigation; its global queue pointer was retired to prevent accidental retry.
The LFM latest-export watcher now uses a new output root and the qualified current
built runtime. Mellum's future-ingest spec uses the same corrected runtime,
while its old7750 historical reports remain unchanged. New requests seal new
runtime inventories; never edit an existing request or turn interrupted unknown
work into negative training data.
