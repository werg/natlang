# Native checkpoint evaluation away from cloud training

Policy: evaluate the newest useful verified LFM export on the spare DGX GPU.
An evaluation already admitted or running finishes; newer exports coalesce while
it runs. Results describe their historical weights and never qualify different
current training weights. Both cloud trainers retain resource ownership.

Transfer continuation correction (2026-10-18): admission starts when the
supervisor commits to downloading a verified export. Pin that exact receipt and
step durably before fetching. Resume that transfer after preemption even when
the exporter has advanced; only coalesce to a newer export after the selected
request drains. The 20513 Mellum transfer was interrupted at 23.9/24.3 GB and
the restarted supervisor began downloading 20515 instead. Its manually rescued
20513 request is now queued; the shared mechanism below prevents recurrence on
either model. This is exact transfer/durability plumbing for the same policy,
with no new teacher, evaluator, quality gate or task limit. The shared service's
disk admission must use the same pinned receipt, including its retained bytes.
The floor callback rechecks that selection under the existing transfer lease;
an export advancing during the preliminary floor check cannot evade admission.
The separate, pre-existing request-publication-to-active-journal crash window
still needs authenticated existing-request recovery; interrupted-fetch tests do
not certify that whole sealing transaction.

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

2026-10-17 storage recovery: actual LFM9088 asynchronous evaluation exhausted
the local disk, truncating journals and then failing to write its failure row.
The exporter also stopped on ENOSPC. Retain that evaluation as unscored; do not
repair its score by ignoring malformed evidence. Restart the existing exporter
and evaluate a newer verified checkpoint in a fresh request. Expose the existing
collector `--min-free-mib` mechanism through the task-evaluation spec (default0,
nonnegative integer). The DGX supervisor selects its existing20GiB free-space
floor for future requests. Before each new case the existing collector records
a storage pause and leaves unstarted cases unscored when below that floor.
This changes only storage admission, not prompts or model outcomes; it cannot
guarantee free space during a case or replace normal storage supervision.

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

## Storage pause continuation (2026-10-17)

A collector's recorded `storage_pause` means its next case has not started.
The evaluator must wait for the declared disk headroom and resume that exact
queue and journal, preserving finished outcomes and the loaded model, rather
than score an incomplete evaluation and delete its snapshot. This is exact
resource plumbing: the existing collector already specifies that recovery.
The helper verifies the unchanged queue, matching pause floor, and unstarted
pause key. Operator drains and other exits retain their existing meaning.
No task, prompt, sampler, model setting, result or scoring rule changes.
