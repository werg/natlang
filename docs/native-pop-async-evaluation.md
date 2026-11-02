# Pop native asynchronous development evaluation

The learner and evaluator may use separate frozen runtimes. Keep the actual
learner runtime and its capacity proof unchanged when only the evaluator changes.
A new evaluator runtime needs an actor transport prepared under that exact
runtime's authenticated Python inventory. An inference view is not a resumable
training checkpoint or independent full-state recovery.

`prepare_pop_async_sft_controls.py` creates a review-only controller config with
`refresh_after_updates=16`, `refresh_at_checkpoint=true`, and
`task_eval_every=0`, together with the exact 16-family development monitor spec
and baseline full-source SHA. It leaves the configured final/foundation task gate
in place. These controls require the actual native controller config; they do not
start a learner. Existing bounded-window logic already skips intermediate inline
task cadence and breaks at the refresh boundary. A completed stage or foundation
boundary can still run its existing synchronous final task gate.

After an exact full checkpoint has been durably published, prepare its actor
transport with the existing `native_bundle actor-transport` and deliver it using
`send_actor_transport.py`. Retain the full source and receiver/consumer lease.
The previously measured full-source actor-view exporter peaked at 19.8 GiB;
reused tensors do not prove its full metadata decode fits beside a learner.
Admit that preparation only with actual memory headroom, or use a separately
qualified lower-memory exporter. Never count a compact view as full recovery.

Inside the Pop evaluator namespace, `seal_pop_async_task_eval.py` accepts the
independently bound view, view receipt, transport, source SHA, frozen monitor
spec, and local paths. It creates an existing native async request and queue
pointer. The queue has one active transport request: a newer candidate stays
pending with its owner and must be submitted again after the active request lands.
Interrupted publication is resumed from the sealed request. The existing
`python -m natlang_neuralese.eval.task_cadence --async-request <directory>`
executes it; the evaluator restores the exact transported actor and records both
the view identity and the original full-source identity. The request snapshot is
separate from the transport cache and is retired only through the existing
completed-request path; a small exact request copy survives alongside results.

Submit the pinned baseline and candidate under the same evaluator runtime,
monitor spec and serving geometry. `compare_pop_async_task_eval.py` consumes
independently SHA-bound task-score and sealed-request files and checks all 16
packet cases, distinct families, source identities, full option/serving config,
Python and task runtime, and actual FP32 BF16 reduction. It reports protected
development outcomes and per-case changes. Missing, pending, incomplete, failed,
or mismatched results do not qualify current weights, satisfy a final gate, or
establish capability lift. Untouched confirmation remains separate.
