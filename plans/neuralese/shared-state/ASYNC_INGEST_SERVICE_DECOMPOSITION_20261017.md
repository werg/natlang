# Continuous current-checkpoint feedback

The supervising policy is to evaluate the latest genuinely verified export as
soon as the preceding evaluation drains, preserving protected cases and useful
ongoing work. A declared feedback-priority dependency may hold ingestion until
another model's exact request is actually executing. This prevents checkpoint
transfers or newly queued long evaluations from delaying that first feedback.
The dependency is historical once genuinely observed; later completion does not
revoke it. Neither ingestion nor evaluation qualifies current cloud weights.

1. Keep one ingestion owner per output directory and the existing sole model
   dispatcher. Use the existing native_async.poll_latest protocol, source
   verification, resumable sequential fetcher and runtime snapshots unchanged.
2. If configured, verify the dependency's active native request, exact saved
   checkpoint/step/architecture and sealed evaluation provenance. Bind its
   dispatcher receipt and unit name derived from the request. Require the unit
   active and a live process in its actual cgroup executing task_cadence with
   that exact --async-request path. Recheck all source bindings before sealing
   durable historical evidence. A download or stale dispatch receipt is not enough.
3. Preserve an existing active request/spec/provenance without rewriting it.
   Coalesce later source exports through poll_latest; require the configured
   role/minimum step and sufficient disk headroom before a new download.
4. Record bounded status, typed errors and latest historical results. Do not log
   private URLs or provider exception text. The shared dispatcher alone starts
   evaluations and their model processes through the ledger.

Natural language specifies priority and continuity policy above. Crisp code is
limited to exact proof checks, process identity, file durability, existing API
orchestration and CLI plumbing. One generic service serves both students; it
introduces no model/sampler/rendering semantics and grants no training exposure.
Tests cover wrong/stale requests, process identity/argv, persistence/restart,
no-transfer priority holds, unchanged active work, and the original fetch argv.

## Transfer memory belongs to transfers (October 18)

The service may use a small monitoring allocation while an already sealed
evaluation runs. It must not reserve a model-sized transfer allocation for that
wait. This applies to both students through the same service.

1. Under the existing ingestion lease, inspect the active request using the
   existing landed-result and drained-request conditions. Pending publication
   requires the normal recovery path; it is never dismissed as an idle wait.
2. When ingestion can advance, journal one worker intent and its exact
   configuration. Run the unchanged poll through the memory ledger with the
   declared transfer budget. The worker performs all existing source, capacity,
   checkpoint, runtime and publication checks. It makes no evaluation changes.
3. Release the transfer allocation when that poll ends. If the supervisor
   restarts, rejoin its exact worker rather than launching a second one. A
   preempted worker is rejoined through its exact ledger requeue and launch
   specification, allowing small polls to backfill behind a large evaluation.
   Only a terminal failure with no queued or launching worker is retried
   through the existing durable transfer/sealing protocol.
4. Keep the same sole evaluation dispatcher and protected task specification.
   Publish only the bound worker result after its unit has stopped. Intent,
   result and status are bounded current files, not accumulating run archives.

Natural-language policy is specified above; crisp additions provide exact
leases, worker/configuration identities, ledger orchestration, durability and
CLI entry points. Existing unconfigured service behavior is preserved.
