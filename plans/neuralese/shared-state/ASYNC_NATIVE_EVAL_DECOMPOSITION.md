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
