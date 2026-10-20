# Live teacher export custody

Owner policy: preserve verified complete trajectories continuously, including repeated samples, before provider loss. Generation, reference qualification, training admission and exposure remain distinct. Protect the live teacher; exports never alter its packets, queues, workers or model.

1. Authenticate the current guardian, actual feeder birth and sealed source batch.
2. Count source cases independently from repeated queue jobs. Verify every job range against the executable packet; preserve every distinct queue key and seed.
3. If the queue uses the existing student-root prepared packet, retain raw source cases, preparation receipt and exact executed packet. Verify the existing student_root_row transformation, both hashes and source paths. Match collector manifests to the executed bytes, not the pre-transform source.
4. List actual worker shard filenames in numeric order using the normal remote Python command. Merge only each shard's authenticated jobs and journal events.
5. Export immutable full trajectories and all source/control provenance through the existing private PUT plus independent full GET verifier. Advance only the durable tranche frontier. Failed, held and unstarted obligations remain explicit; no export grants admission.

Python is crisp SDK/file/hash/schema/range/transform verification and atomic durability plumbing. Source selection and admission policy are unchanged. The unchanged pure student_root_row function is factored into common/task_roots.py and remains imported by task_rollouts; this avoids importing the unrelated Torch evaluation harness in the CPU exporter. Its AST/body and behavior are retained. The shared helper owns the transformation; the exporter does not invent another renderer or teaching prompt.

For every selected queue job, fetch its own generation-held sidecars before sealing the tranche. The verifier requires each distinct program/index and exact bytes; held jobs remain held, never positive trajectories. This is custody plumbing, with no generation or admission policy change.

Closed boundaries apply the same exact raw-source/student-root preparation verifier as live tranches. Only its authenticated paired packet/preparation are canonical control files; arbitrary extras and missing repair sidecars still fail. Closing a batch never converts pending or held work into success.

Create the owned private-receipt directory before invoking the existing exporter: its upload-phase and verification receipts are durable children of that directory. A successful PUT without the subsequent full GET never advances the frontier.

After a successful scoped PUT, notify the existing W&B SDK service that this
exact run file was uploaded when the server supports `MARK_RUN_FILES_UPLOADED`.
Object-store notifications can fail to register otherwise valid bytes. The
shared export guard supplies this service mechanism for teacher and student
exports; it neither resumes the run nor grants verification. Continue to require
the independently downloaded complete object's original SHA-256 and size before
publishing any durability receipt or advancing a frontier.

When recovering a sealed tranche whose private upload previously failed, prune
only the returned accepted tranche after its receipt and frontier are durable.
The pre-upload in-memory tranche still has no private receipt and cannot authorize
cleanup. A restart after the frontier commit verifies the saved accepted manifest.

Idle flushing measures absence of new eligible completed outputs, independently of activity heartbeats from still-running workers. Keep the complete journal SHA as telemetry. Both existing publisher loops use one exact completed-output identity helper; completion/accounting validation and every repeated queue job remain unchanged.

Canonical guardian repair queues namespace original jobs as
`batch-0021:s7-task-069-00`. Preserve that exact key in queue, journal and tranche
receipts. A colon is a literal safe POSIX filename character inside the existing
`outputs/<ordinal>-<key>` prefix; permit it without accepting separators,
backslashes, whitespace or escape characters. This is filename custody only,
not a new sample or repair-success classification. A separate existing publisher
frontier can export the bound repair attempt while the next main batch runs.

# Natural feeder exit during publisher discovery (2026-10-18)

The publisher must follow a genuine batch transition without confusing a
missing old `/proc` entry with a changed actor. The guardian remains the sole
generation authority; the publisher does not start or stop collectors.

1. Verify the existing boot, config revision, state and exact feeder birth.
2. If the old process has disappeared or exited and the exact batch phase is
   complete, keep polling in non-exporting draining state while the guardian
   commits its boundary. There is no arbitrary closure deadline.
3. Follow advanced custody only after the same authenticated config/state names
   an advanced batch and the exact old boundary bytes are bound and verified.
4. Keep unchanged wrong births, boots, config revisions and unbound transitions
   fatal. Export no active rows from the stale snapshot. The ordinary boundary
   exporter still reconstructs all completion accounting before publication.

These are crisp process-custody and immutable-byte verifiers. There is no new
generation selection policy, prompt, actor identity field, or training admission.

## Natural feeder preparation and completed-phase grace

A valid live feeder may be between its durable process launch and publication of the collecting worker table, or between publication of its completed phase and guardian closure. Treat only those exact known phases as a non-exporting wait after rechecking the same boot/config/process ownership. Keep all source/worker/runtime validation for collecting phases. A different live birth, unknown phase, changed config or unauthenticated advancement remains fatal. This is custody plumbing; no model-facing behavior, generation, admission or timeout policy changes.

## Empty trailing worker shards (2026-10-18)

Policy: retain every genuine completed teacher job, including batches smaller
than the existing worker count. An unused shard neither grants completion nor
contributes a job.

1. Authenticate the nonempty master queue and every numbered worker queue.
2. Permit empty worker queues; still require the exact unique contiguous shard
   inventory and the union of nonempty shards to equal the master exactly.
3. Accept journal events only for keys owned by that exact shard. Thus an empty
   shard cannot supply any event or borrow another worker's completed job.

Queue and journal byte verification is crisp custody plumbing; no model-facing
text, scheduling policy, source selection, or training admission changes.

## Virtual boot identity transport (2026-10-18)

Authenticate the actual kernel boot on every existing transition check. The SDK
file service may return empty bytes for zero-size virtual proc files. Reuse the
existing bounded, read-only `cat` exec transport for the exact boot-ID path in
the scoped reader, including multi-read callers. Ordinary artifact paths retain
the SDK file transport. Nonzero exec, empty identity, changed identity and
transport failures remain failures; no stored boot value substitutes for a read.
This is crisp outside-world transport, with no policy or model-facing change.


### Whole-poll telemetry and discovery failure custody (2026-10-20)

The supervisor diagnoses slow export transport separately from healthy generation.
The crisp owner-host publisher records start, durable completion and monotonic
elapsed time for the whole poll, including overlapping batches. A primary
batch result timestamp remains that batch's observation, not a whole-poll
heartbeat. Read and validate the existing closed-boundary receipt map before
SDK discovery, so a discovery failure after restart cannot erase earned
publication custody. Neither timing nor retained receipts grant admission.
