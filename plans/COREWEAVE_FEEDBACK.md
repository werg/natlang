# CoreWeave sandbox feedback log

Usability problems and surprises we hit with CoreWeave GPU sandboxes (`cwsandbox` SDK/CLI, W&B-backed accounts and
storage), kept so we can give the CoreWeave/W&B team concrete feedback. Append every new one with its date, what
happened, impact, and whether the cause is confirmed to be on the platform side or possibly ours. Our own bugs
(closure shipping, bootstrap scripts) are excluded unless a platform behaviour made them much worse.

Format: **date: title.** what happened · impact · attribution · suggestion.

## Capacity, shapes and errors

- **2026-10-29: unspecified CPU/RAM silently becomes 1 CPU / 16 Gi.** With CPU and RAM left unspecified on a one-GPU
  (H100) request, the sandbox got requests = limits = 1 CPU and 16 Gi. · A cohosted coordinator starved and fenced,
  and workers OOMed on catch-up. · Platform default (confirmed via `cpu.max`). · Scale the default with the GPU, or
  warn at create time that a GPU sandbox got 1 CPU.
- **2026-10-23: one-GPU request granted a 96 GB GPU with 2 Gi host RAM.** · Unusable for model loading; closed
  unused. · Platform placement. · Enforce a sane host-RAM floor per GPU, or report the granted shape prominently.
- **2026-11-03: `SandboxResourceExhausted` doesn't say which resource.** It reads like an account quota error, but it
  means the cluster can't currently fit the requested CPU/RAM shape. · We misdiagnosed it as exhausted accounts and
  wasted retries. · Platform message. · Name the constrained resource and say whether it's transient capacity or a
  quota. A "largest currently schedulable shape" hint would help even more.
- **2026-11-04: grants land at the bottom of the shape ladder.** d4 (H100) was granted only at our floor of
  5 CPU / 32 Gi after stepping down from 8 CPU / 64 Gi. · Less host RAM than other learners (risk with offload off). ·
  Capacity, probably. · Same as above: a shape hint at refusal time.
- **Transient control-plane answers** ("being reconciled with its runner", deadline exceeded) on reads, so all our
  reads retry for up to 15 min. · Slow and noisy tooling. · Platform. · Documented retry semantics or idempotent
  status reads.

## Lifecycle and reliability

- **2026-10-28: sandbox lifetime is tied to the bootstrap exec's output collection.** Restarting a learner as a
  detached SDK exec doesn't replace the original bootstrap child. When its output-collection timeout expired, the
  sandbox completed and killed the replacement processes and uploads, despite a 720-hour lifetime. · Lost work;
  forced us to build a PID-1 keepalive and driver-ownership layer. · Platform semantics (confirmed). · Decouple the
  sandbox lifetime from the bootstrap command, or document it prominently and offer a "keep running until stopped"
  mode.
- **2026-10-26: same-ID workspace resets.** A sandbox stayed `Running` under the same ID but restarted with a new
  boot identity and an empty workspace. · Unexported state lost; the old progress reading looked live. · Platform. ·
  Surface restarts/reschedules as an event and change state, or at least expose a boot ID in the API.
- **2026-11-04: learner sandbox b4caeaa0 died around 06:13Z without a stop request.** · 13 earned steps lost. ·
  Unconfirmed: it may be the lifecycle behaviour above or our bootstrap. · An exit reason or event log for every
  terminated sandbox.
- **2026-10-28: completion status contradicts the process exit.** The provider reported exit 0 while an upload
  process had died with 137. · We nearly misdiagnosed an OOM. · Platform reporting. · Report per-process exit codes
  or OOM-kill events.

- **2026-11-04: sandboxes complete with no reason.** On 2026-11-04 every GPU sandbox ended in `completed` with an
  empty RETURNCODE and no event or exit reason: d1 fea754f2 (cw2, running about 11.6 h), d4 66644e8f (cw2, about 1.7 h),
  bankmeta 17382ab4 (cw1, about 18 h), and earlier w0/d2/d3 (cw4, around 11:00Z). For w0/d2/d3 we know of one cause on our
  side (a driver that wasn't registered with the bootstrap); the others are unexplained. · All production training
  stopped, and the DiLoCo coordinator state on d1 was lost. · Attribution unconfirmed. · A termination reason (who or
  what ended it: entry exit, OOM, preemption, node drain, quota) on every completed sandbox.

## Recovery, snapshots and storage

- **2026-10-23: no recovery from a quota cutoff.** Both normal and gateway file reads reject a completed sandbox, and
  a stale `Running` container field doesn't grant access. Snapshot requests after shutdown failed because no snapshot
  storage had been configured, so unexported state needs provider-side recovery. · Lost state on the quota incident.
  · Platform. · Allow read-only file retrieval from completed sandboxes for a grace period. Warn at create time when
  no snapshot storage is attached.
- **Snapshots are account-scoped**, so with four accounts the DGX must act as the hub for cross-account restore. ·
  Extra transfer work. · Platform design. · Cross-account snapshot sharing within one owner.
- **Storage deletions leave zero-byte tombstones** that need a fresh HTTP 404 check to confirm deletion. · Extra
  verification logic in retention. · W&B storage. · Clear deletion semantics.

## Quotas, credit and accounts

- **2026-10-29: hard quota cutoff with no warning** stopped all learners, the teacher and both evaluators
  mid-work. · Lost unexported work. · Platform. · Warnings at thresholds and a grace period to export.
- **Opaque credit:** no visible balance, rates or per-account spend, and the trial quota is a lifetime cap. · We can
  only budget in GPU-hours we count ourselves. · Platform. · Expose balance, rates and usage per account.

## Networking and services

- **2026-11-04: no inbound service port on a running sandbox.** A port must be declared at create time
  (`service: true`), so we couldn't add a DiLoCo coordinator replica peer to existing learners. · Delayed coordinator
  redundancy. · Platform. · Allow exposing a port on a running sandbox.
- **Signed-URL transfer throughput:** each stream from W&B storage ran at about 12 MB/s. Eight parallel streams also
  totalled about 12 MB/s into the DGX, so the DGX link may be the limit. · Exports and seeding take tens of minutes
  per checkpoint. · **Unconfirmed**; measure sandbox↔storage separately before reporting.

## Tooling

- **No CLI file copy into a running sandbox**: pushing files means SDK `write_file` from a script (or `exec` with an
  inline payload). · Ad hoc scripts for every repair. · Platform CLI. · `cwsandbox cp`.
- **The granted resources aren't returned in a convenient place**: we exec into each sandbox to read `cpu.max`,
  memory and GPU. · Platform. · Return the actual granted shape in the create/get response.
