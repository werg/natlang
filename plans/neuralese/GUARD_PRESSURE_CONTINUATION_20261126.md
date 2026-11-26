# Guard pressure continuation inventory

Actual Source54 adaptive job: guard stopped at7,453,749,248B working versus5,905,580,032B claim, available6,362,628,096B. Reason was `over budget under memory pressure`. Existing guard deliberately queued only preemption/free-floor victims, so no waiting entry was created. A cooperative exit0 did not erase a queue entry. No retrospective launch of that underestimated profile is authorized.

- memory_ledger.py victim/guard: all DGX/Pop local CPU/GPU processes share this pressure allocator. Resumable pressure overshoot must retain a durable queue entry using at least observed working use times1.1; --no-resumable/services keep their existing protection. Existing backoff/reserve/headroom/container-join rules still apply.
- requeue_spec/claim_for/launch: an already-raised claim previously retained stale spec.budget, reverting on relaunch. Queue admission and stored budget must agree. Preserve original effective MemoryMax and CUDA execution allowance; raising accounting is not permission to change the computation or hard limit.
- run/cancel: existing cancellation generations distinguish a withdrawn continuation from fresh admission. Never resurrect a cancelled queue entry; active execution is not terminated by cancel.
- CollectorCheckpoint (bulk and Android tail): normal completion and explicit owner signal without guard-owned continuation disable future continuation through existing cancel API. A guard-owned queued checkpoint remains resumable. Canonical stop-after-wave/runtime hold must override a concurrent guard queue.
- vibestudio_campaign.drive: preserve exit0/cooperative retained data. Explicit stop-after-wave wins; guard-owned pressure checkpoint is continued by allocator, not a second controller.
- wait_exit: already restricted to --no-resumable (run rejects otherwise), so not involved and unchanged.
- Colab admission.py lines1–18: separate discrete-VRAM/host tenant eviction, no memory_ledger victim/requeue spec or free-floor guard. No change to VM leases, actor/learner/optimizer/RNG. Engine implementations (llama-server/wllama/Python/SGLang/ts-host) do not implement this out-of-process allocator; their command, hard caps and execution allowance remain unchanged.

No production profile, existing unit, source freeze or current ledger file is edited by source tests. Explicit job opt-out remains --no-resumable. Current54 profile correction is owned separately.
