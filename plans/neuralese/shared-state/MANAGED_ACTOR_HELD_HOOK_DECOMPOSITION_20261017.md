# Managed actor held monitoring

The recipe decides which protected evaluation to run. Exact lifecycle plumbing
must run that declared command only after the managed actor reports the exact
current checkpoint, synchronously before TRAIN collection, keeping the same
actor alive. Extend the existing run_command hook mechanism with optional
`after_actor_ready`; do not move or reinterpret before_collect. Preserve this
command with the sealed actor/task configuration during unfinished rounds.

The held command uses separate protected queues, journals and outputs and never
publishes training rollouts. Failure disposition remains explicit in its own
monitoring receipt. Actor lease cleanup still runs on hook/collection errors;
completed sealed collections skip both actor creation and hook. No model copy,
parameter update, sampler rewrite or historical rollout relabeling is involved.

A numerical runtime stays immutable. A provider may declare one exact
`worker_entry` file binding under the existing native-production namespace.
Verify its full bytes/SHA on the target before launching it; pass the expected
SHA to the entry so it self-verifies before loading the declared CPU control
composition and dispatching the unchanged native bridge. The detached runner,
request identity, launch lease and default argv stay unchanged. Numerical actor
and trainer subprocesses retain their declared original runtime. This extends
the existing bound operational entry mechanism, not numerical code inheritance.

The monitoring CLI reuses TaskEval queues, collector arguments, exact scorer and
storage-pause continuation with an explicit existing actor URL. It authenticates
current checkpoint/native policy/rendering before and after, binds the task spec,
and uses a separate held-only output lease. Completed monitoring is idempotent;
interrupted external evaluations preserve existing queues/journals and resume
finished keys. Default trainer-owned engine lifecycle stays unchanged. Collector
failures are recorded as ineligible monitoring, never invented task successes or
TRAIN rows. The managed actor remains alive throughout this synchronous command.
