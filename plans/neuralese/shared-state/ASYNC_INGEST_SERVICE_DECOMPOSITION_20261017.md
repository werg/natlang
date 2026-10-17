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
