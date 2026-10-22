# Native asynchronous checkpoint decomposition

## Goal

Reduce accelerator idle time at ordinary native SFT checkpoint points by
persisting an immutable full-state snapshot while the next training updates
run. Preserve exact model, optimizer, RNG, exposure, schedule and receipt state.
The ordinary atomic checkpoint remains the recovery authority. A failed async
write must be surfaced before another optimizer update or handled by an exact
post-commit recovery save; it must never claim a newer parent was preserved
when only an older checkpoint is durable.

## Policy and mechanism

The supervisor's natural-language policy decides whether a checkpoint may be
overlapped with further training. It permits overlap only at ordinary cadence
boundaries with a detached snapshot, sufficient measured host/cgroup headroom,
one writer slot available, and no immediate consumer of the saved path. It
requires a synchronous flush before task evaluation binds an exact checkpoint,
before a final/stage-completion boundary returns, and when a stop signal is
handled. It halts updates when writer failure is observed and applies the
existing post-commit recovery policy to the actual in-memory committed state.
This policy is configuration-driven and shared by LFM and Mellum.

The crisp mechanism is responsible for exact tensor snapshots, thread custody,
atomic file durability, memory measurement, writer errors, and exact checkpoint
publication. It reuses `immutable_cpu_snapshot`,
`AsyncAtomicCheckpointWriter`, `available_system_memory_bytes`, and
`atomic_checkpoint`; it does not introduce a second serializer or weaker save
path.

## Decomposition

1. **Choose an ordinary cadence boundary** — reuse `CheckpointPolicy` cadence
   and its existing `save` call. A configurable setting selects async only for
   ordinary cadence writes; mandatory consumers and boundaries force a drain.
2. **Admit the host snapshot** — compute a conservative serialized-state bound
   including uninitialized optimizer slots, compare it with live
   `MemAvailable` and cgroup availability using the warmup caller's safety
   margin, and fall back to synchronous persistence if unavailable or
   insufficient. Record the decision and measured headroom in checkpoint
   events.
3. **Freeze exact committed state** — construct model/optimizer/RNG/exposure
   and training metadata at the committed update boundary; copy tensor storage
   to CPU with alias preservation. Submit at most one detached snapshot and
   drain it before creating another.
4. **Persist and publish** — reuse the existing atomic pending-file, fsync,
   rename and post-write offload sequence. Run checkpoint publication only
   after the exact file is durable, using the matching detached snapshot;
   report publication receipts only after their durable status is written.
5. **Guard subsequent updates** — check writer completion/failure before each
   optimizer update. On failure, stop advancing training and invoke an explicit
   exact-current-state recovery save; if it cannot commit, fail with the last
   verified durable checkpoint identified and no false preservation claim.
6. **Flush consumers and exits** — drain before task-evaluation hash/load,
   publication consumers, final/stage completion, stop handling, and process
   exit. Preserve synchronous semantics wherever a consumer needs the just-
   saved path immediately.

## Validation and status

Validation must compare detached checkpoint trees against the exact source
state, including storage aliasing, model tensors, optimizer slots, RNG and
exposure for both model families. Integration checks must hold the writer open
while a real optimizer update occurs, exercise synchronous fallbacks and
consumer flushes, and inject writer/post-write failures. Host-memory eligibility
must be measured on the actual assigned machine before async is enabled.

Implementation is now wired through the shared `CheckpointPolicy` and
`native_state_sft`: `--async-checkpoint` is an operational recipe/resume
setting, defaulting off. The policy admits only ordinary cadence writes after
the conservative state-plus-lazy-optimizer host/cgroup check, drains any prior
writer before cloning, and runs the existing atomic/offload path plus optional
publication callback on the immutable snapshot. Task-evaluation checkpoints,
stage ends, refresh/stop boundaries and final writes remain synchronous. The
trainer checks completed writer errors at each update boundary and saves the
exact current full state synchronously before moving on; it drains before
returning at stage/stop boundaries. Exceptional trainer exits also drain in a
guaranteed path, retain the primary training/close exception, and attach writer
or recovery failures as secondary notes. They save a recovery snapshot only at
a known completed-update boundary; a failure inside the optimizer update is
drained and reported without serializing possibly partial model/optimizer state.

Focused ledger validation exercised actual native LFM and Mellum optimizer
updates while a writer was held, exact model/optimizer/RNG/exposure snapshot
content and alias preservation, insufficient-memory synchronous fallback,
writer-failure recovery, forced synchronous saves, a trainer exception plus
failing `schedule.close()` with an outstanding writer, and mutable metadata
isolation, plus the shared writer and checkpoint-policy regression suites.
Local implementation tests do not
establish safe host headroom on the production sandbox; the setting remains
default-off until the supervisor measures it and explicitly enables it.
