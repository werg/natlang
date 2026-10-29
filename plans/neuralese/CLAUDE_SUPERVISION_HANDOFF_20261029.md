# Training supervision handoff to Claude Code — 2026-10-29

Owner requested a convenient, deliberate handoff from Codex to Claude Code.
This is an operational handoff, **not completion of model training**. Read
`AGENTS.md` first. The final observed state is recorded below; acceptance of a
role or a provider Running label never establishes an optimizer update.

## Immediate priorities

1. Finish the **one four-worker sharded DiLoCo cohort** and verify actual updates
   on every worker plus a real outer merge. Never launch four duplicate standalone
   learners. Supervise host pressure, GPU use, actual data exposure and recovery.
2. Use the corrected own-output loss weights and complete current Kahan runtime.
   The old suspect lineage stays diagnostic. Preserve the fresh step202 parent
   and verified recovery until the new cohort's useful replacement is durable.
3. Admit the ready Big2 additions and full-prompt teacher variants promptly through
   explicit resident continuations. Do not confuse prepared data with exposure.
4. Measure actual task behavior early; alternate useful SFT with full-prompt,
   on-policy whole-graph JOLT+. Fix concrete runtime/custody bugs before admitting
   affected episodes. Program adaptation follows a meaningfully trained backbone.

## Owner direction that matters now

- Best capable models promptly, adaptive runs, all useful data and the entire
  evolving recipe; epochs and stages are not sacred. Finish an in-flight optimizer
  update, checkpoint and change the run whenever justified. No epoch wait is needed.
- LFM3 is the current focus. Its joint SFT trains natlang execution and native
  Neuralese together. It has no dependency on Mellum or cross-model VAE work.
  Mellum artifacts remain useful recovery; do not restart an unrelated cloud line
  automatically while this four-worker placement is being completed.
- No additional teacher generation for this LFM3 lane. Existing teacher data is
  useful with original actor/source/split custody. Static rendering workers are
  not Luna generation. Historical instructions about six Luna generators are
  superseded by the later LFM3 direction.
- Reward correct tasks with or without communication. When communication exists,
  update the entire actual producer/consumer graph, including hybrid state and
  exact occurrence bindings. Never detach an in-episode producer for convenience.
- Changes ship when useful and ready, including continuations; do not freeze a
  worse setup merely because a run began. Preserve useful optimizer/RNG/exposure
  and exact transition history. Base-relative KL is telemetry, not a stop rule.
- Main is the working branch. Integrate valuable orphan work; do not abandon it
  in worktrees. Preserve unowned WIP and live recovery dependencies.

## Four sandbox identities and topology

| Rank | Sandbox | Assigned resources | Role |
| --- | --- | --- | --- |
| 0 | `f3ad785f-9f0b-4716-905a-ed26f0a83e1e` | H100 80GB, 16GiB host, 1 CPU | learner plus primary coordinator |
| 1 | `ea17666b-084f-4236-b552-48efd43f9f74` | RTX 96GB, 16GiB host, 1 CPU | learner |
| 2 | `3aba3399-4a8d-4e48-b8a9-069e653bf0ef` | RTX 96GB, 16GiB host, 1 CPU | learner |
| 3 | `b1662ef1-2514-49f4-be56-645e83a6712c` | RTX 96GB, 16GiB host, 1 CPU | learner plus replicated coordinator peer |

The primary and replica endpoints are the provider's actual exposed port8080
URLs. They require the scoped bearer and gateway credentials delivered in private
files; do not put token values in handoff text, logs, command lines or git.
Gateway 503 persisted temporarily even after the replica listened locally and
answered authenticated requests. Verify local listener and exact provider URL;
do not assume the process died or restart blindly after a short observation timeout.

All roles use persistent owner supervision. Original bootstrap collection timeout
can otherwise complete the sandbox and kill detached replacements despite a long
provider lifetime. Use exact owner/PID/start/boot and generation guards. New GPU
requests require one GPU; do not reintroduce GPU-model/CPU filters. Current
allocations already exist: do not spin up another merely for routine evaluation.

## Current launch: v4 / 969d

Authoritative control directory:
`/home/werg/data/dgxops/claude-recovery-20261028/data/fresh-first-outer-step202-20261029/controls-main-v4`

- Runtime is the **complete committed native package at `969d35b7b`**, not the
  earlier special b0 patch tree. Runtime archive SHA
  `4a450e95902318981be2710d4920862968f22d4effd4b1cc593dc0f0f21a9fd2`.
  Archive has 399 Git blobs; its native inventory has 369 files. All four hosts
  verified that inventory and the shared static argument checks.
- Control archive `control-transfer-main-v4.tar.gz`, SHA
  `c8d160f663d305f066198de96866ace12319238f3196ecb5ff89843942709c10`,
  thirteen individually verified members. Paths in its worker args, plan,
  role-contract, peer declarations and coordinator args are authoritative.
- Explicit **Kahan rounding**, **local weight EMA off**, accumulation16,
  local_steps32, quorum3, grace30, BF16 wire. Global coordinator EMA remains on.
  Initial ten outer rounds average; later Nesterov uses momentum0.9/lr0.7.
- Rank0 resumes its exact earned step202. Other ranks clone exact common step2.
  The first outer mechanism preserves rank0's newer weight deviation and RNG,
  and gives seed clones deterministic distinct rank RNG streams. Initializer
  and reanchor are not optimizer progress.
- Runtime path on hosts is `/work/runtime-four-sft-969d-v2`. New output and
  coordinator namespaces are v2; consult the v4 args instead of guessing paths.
- At 13:45–13:46 Berlin, primary init PID83047 and replica init PID80511 had
  been accepted; actual completion, new baseline, bootstrap, updates and merge
  still required observation. Final handoff observations below supersede this
  intermediate timestamp.

Operational launch/probe scripts live in:
`/home/werg/data/dgxops/four-sft-cutover-20261029/`.
Use `*-v4.py`, `*-v2.py` and current receipts; old similarly named attempts are
historical. Read a script before executing it: some start/stop roles. Routine
health uses actual progress and authenticated manifest, since full status can
wait behind large tensor merges.

## What went wrong during cutover and what was fixed

1. The original production coordinator init mapped the parent but omitted
   `MappedInputEvictor`. `b0a52cae8` wires consumed clean-page release into bounded
   serialization. Seven actual mmap/serialization tests passed.
2. Generated coordinator args omitted grace30, inheriting8. `4c4c60b8e` fixes it.
3. Three hosts lacked nine Qwen36 source-evidence dependencies. A single 13.3MB
   compressed / 59.1MB raw exact repair was staged. All four then verified every
   SHA in the complete 635-path role closure, not just file presence.
4. The old row-count CE scales would give coding about49% of own-output mass.
   The completed source-bound census enabled an explicit corrected objective
   before the first distributed update. No distributed update used the old policy.
5. V3 learner argv retained local weight EMA while DiLoCo owns global EMA. All
   four exited before model load or SGD. The trainer already supports recorded
   `local-ema-retired-for-coordinator-ema`; v4 explicitly uses `--weight-ema off`.
   `969d35b7b` extracts the actual trainer's static semantic checks and invokes
   them in the builder, fixing the parse-only preflight gap.
6. Earlier replica CLI only accepted an inline gateway secret. `f3f0d4a6f` adds
   private gateway-file support. The old bootstrap used a private0600 ephemeral
   config; its exact processes were fenced before v4. Do not reuse that baseline.
7. Kahan is now complete in the selected package: rounding-mode changes actually
   clear stale compensation; fragment installs retain the local residual plus
   FP32-to-BF16 install error; real authenticated drained role changes are
   implemented. Relevant commits `cbd168d39`, `8cf7c6a71`, `6cc701690`.
   Six real transition tests and 28 protocol/heterogeneous checks passed.
   Production memory/throughput still needs actual update evidence.

The previous 88-fragment / 23.47GB baseline had verified byte-identically on both
hosts, but its trainer identity used stochastic/local-EMA controls. It is not the
v4 baseline. Seed/fragments can be reconstructed locally on the replica and
verified against **every file hash in the primary receipt**, avoiding a 23GB
gateway transfer. Never substitute a separately captured receipt with different
timestamps or claim equality from fingerprint alone.

## Recovery: concrete copies, not acknowledgements

Common step2 full state: SHA
`51bacc47b54e937684e8e757dde12e1b01719c8f24e1d52d8f185b236753648f`,
12,482,163,595 bytes, present on all four at
`/work/snapshot-recovery-control/fresh-common-step2-51bacc47.pt`.
Private PUT/cloud GET and independent workstation full GET/hash were verified.
That initial owner GET was streamed, not retained locally.

Fresh step202 full pin on peer1: SHA
`fd28c2213c4aebdd0489dfc1f19cb27b312a4de424215e24ad0577370c757bc4`,
12,482,163,723 bytes. Same boot/PID/start continuity from actual fresh writer,
optimizer/RNG/exposure/source and final report verified. Private PUT plus
independent cloud GET/hash completed; newer owner GET must be tracked separately.
The sealed first-outer inventory contains the exact pin and witness bindings.

**Retained local recovery is qualified at step142:**
`/mnt/external/natlang-training-recovery/lfm3-fresh-20261029/peer1-step142/checkpoint.pt`,
SHA `bf13a50b827e034fcf1fa5c0d66c87031787472a32a11761419dd0ef04a7d961`,
12,482,163,723 bytes. Full owner GET/hash and subsequent mmap metadata confirmed
optimizer/RNG/exposure/SFT progress, train9d931 and held5053. The download used a
hidden temporary file; an empty directory before atomic commit was not failure.

Recovery for the live cohort must cover accepted coordinator queues, sequence,
session/source transitions, outer momentum/EMA and every rank's full inner state.
Replication is supplemental, not quota-independent export. Track actual private
object presence and owner GETs; keep latest+verified fallback, bounded queues and
retention. Do not let old standalone retention remove the fresh lineage.
Old publication watchers were inspected and contain no learner restart path;
preserve useful pending historical exports until their dependencies are resolved.
**At startup there was no configured v4 publisher or v4 health watcher.** Old
standalone watchers are not coverage for the new output directory. Establish
current-owner publication and monitor actual commits/exports promptly. Any
remaining gap is an immediate receiving-supervisor obligation, not qualified
automatic recovery.

## Data and objective state

Initial source has **591,181 rows** (not617k), SHA
`9d93141df15287e147afb7fa32b499ef4d59d54cdfe5de1755d24b879a821362`.
Protected held SHA
`5053ab2b6754f2998294215e7892b5c18443d973f55e6c04239803ffac7b5e7f`.
Full and foundation named roles move together. Keep separately named predecessor
and held inputs exact; do not replay already-consumed source corrections.

The exact9d93 census is
`/home/werg/data/lfm3-sft-20261029-v6/census-9d93.json` SHA
`f180aabf2d7a2e681fc1d6ff68ef658ee0b6233457a9ffcc3a3ea8eabeeaae9e`;
per-row masses are in `census-9d93.rows.json` SHA
`c1b74e087cfd750c3306ea901ab5ce861f71719f80d28037584a828b72084926`.
All519 leaf hashes were checked; no row has zero own-output mass.

| Existing group | Own-output mass | Corrected scale |
| --- | ---: | ---: |
| Communication | 46,172,220 | 0.8711904170949545 |
| Coding | 16,026,694 | 0.9411983750360492 |
| Ordinary | 38,363,075 | 1.179595093719677 |

These give40/15/45 weighted mass. They are explicit family coefficients under
`declared/1`, not the earlier row-count normalization. Candidate cc17 required
an added exact predecessor declaration; current derived policy is `corrected-ce.json`
in the first-outer directory, SHA prefixdbf48784, objective63d779, transitioning
from786931. Preserve its full exact binding from v4 controls.

Ready Big2-001/002 add226,019 complete rows, with archive-backed custody and no
ID or full-content collisions against current train/held. Successor has817,200
rows, SHA `8eace17178fc3202490618a354ac8aeeb6170225c3176e24e620a4a96b9e2058`.
The first incomplete519-leaf CE candidate is invalid: successor needs521 leaves.
Final v4-rebased successor recipe SHA
`4cf011f1c2948602ceec0182313703b1274d2e29f468ff8bc4ba3c6ba3de2224`,
CE SHA `129daeba201e23c48c2b5f80b27796d81e3e232cfe686cb4bf19d29011871175`.
Actual resident contract/source/map/CE preflight passed; it is **not yet published,
adopted or exposed**. Transfer:
`.../prompt-variants-pilot-v1/big2-first-outer-resident-transfer-v2/transfer.tar.gz`,
SHA `f7e9d922a730cdb40e17ce897ba42c331f0050c02e855a94d20016d7ce96a09e`.
Use committed resident payload geometry, fixed by `676481dba`; stage all exact
dependencies, publish against each actual predecessor, verify each adoption and
subsequent exposure. Do not overwrite the distributed/reanchor controls.

Claude owns full-prompt teacher variants, last report255,592 written /249,155
with identical own-output mass to their header twin. Root requested classification
and target-contract verification of the6,437 differences; do not assume all equal.
Append to817200, preserve useful header variants, authenticate variant IDs and
related-world splits. Four-family target40% communication/30% curated natlang/
15% coding/15% other ordinary follows authenticated row reassignment proof, not
a map-extension claim that silently relabels old rows.

Remaining supply: re-encode historical12,413 whole DPO pairs for LFM3, including
425 long pairs; old LFM2.5 tokenized files are not ready native LFM3 inputs.
Complete long-chain sources and physical geometry/custody remain obligations.
Do not count prefix rows or registration as full communication-graph exposure.

## Evaluation, RL and regression evidence

The old suspect weights really regressed on matched execution: instruction base
17/23 versus oldstep12957 3/23; one world per16families gives10/16 versus3/16.
See `LFM3_REGRESSION_INVESTIGATION_20261029.md`. Suspect state is diagnostic,
never a production resume parent. Loss decreasing was not capability evidence.

Claude's latest base gate was33/38 with full prompts,150request/turn cap and
3600seconds, on the protected development packet. Its chosen noise threshold31
is a diagnostic convention, not proof of lift. New harness/error-text changes
need matched measurement; debt is recorded in `MODEL_FACING_CHANGES.md`.

Native step33 cloud evaluation was stopped to free rank3. Partial result:
four correct, two semantic mismatches, two unresolved by supervisor stop. There
was no matched nativebase0 run. Do not call this fresh-lineage regression/lift.
An independently downloaded model-only view is now local:
`/home/werg/data/lfm3-eval/fresh-step33-native/view.pt`, SHA
`5d61173d2de52482ff826dc3dad6e1fdaefc9a5569b2168b8779c38d5ab31973`,
3,457,806,715 bytes, full owner GET/hash completed13:44:57 Berlin. This is NOT
a resumable full checkpoint. The two exact transport/view receipts are adjacent.
Original full source SHA03a95b70…; diagnostic transport scope is
`diagnostic-local-pin`. Transport attestation binds the exact380-file Python
runtime digest72598923… from e3fe7d762; current-main package differs. Use a
verified matching reader or a legitimate re-attestation from the full parent;
do not bypass runtime binding to make it load. No DGX actor was started by root.
Prefer local evaluation/bursts instead of occupying a cloud training GPU.
The retained full step142 is another practical local actor source and avoids
step33's frozen-reader transport issue: use its exact full checkpoint and the
pinned local instruction-base snapshot with the current native loader. It still
needs actual serving checks and matched task execution.

The base0 JOLT diagnostic completed one actual update on a separate checkpoint
SHA `0d13455a1fde4f4c77256f2fbf1efbfdbf8cb2b2841b3ef7ec9a4cb03d908aa5`.
All three task executions succeeded, advantages were zero and all graphs had no
communication edges. Its nonzero loss is inferred to be privileged teacher KL;
the old update lacked per-term saved telemetry. Do not claim outcome-gradient
lift or Neuralese communication learning. Four unused marker rows had~6.4e-12
gradient norms; their unchanged values were consistent with tiny signal under
warmup/epsilon, not demonstrated optimizer misrouting. Reasoning rows changed.
Actual per-term telemetry is now saved by cd04e7779.

Future JOLT has a reported imported-native-state/guidance insertion-coordinate
bug. Claude's helper owns a failing test/fix on main. Repair before admitting
affected communication episodes. It does not block correct ordinary SFT. Newly
integrated service/iteration/greenhouse oracles need exact source-bound task
qualification; do not re-use old proof hashes after changing reward semantics.

## Tests, repository and stranded work

- `scripts/check-main.sh --quick` must pass before push. Last quick failure was
  two newly integrated chain/pair scripts with machine paths; `624b9a67c` fixes
  them. Re-run the quick check before final publication.
- Full current-main ledger is `natlang-check-main-four-current-1029`; log
  `/home/werg/data/dgxops/check-main-four-current-1029.log`, TMPDIR=/tmp.
  It is live, not an all-green result. Root Python phase:3070pass/10fail/5skip;
  all ten failures were old reviewedOpenRouter/Luna launcher tests collected
  before Sonnet removed both tests and implementation during the run. Do not
  restore removed legacy infrastructure. Native suite was~57% at13:46; inspect
  its current terminal result and remaining GPU/TS phases.
- A semantic-stage test failed because it expected a deliberately removed
  cross-model role exchange. `62c4884f2` preserves real frozen-preparation checks
  and asserts the explicit rejection; its targeted test passed. This is not a
  production model regression. Keep any other full-run failures explicit.
- The older frozenbc319 suite was stopped after its useful failure evidence was
  acted on. HDD-local temp fixtures caused path-custody/time failures; current
  main targeted assembler+HTTP/recovery/replica checks passed28 on internal temp.
  Exact model/HTTP CE crash-resume test `bdac60de7` passed. New helper suites and
  their actual coverage are distinct from whole-main qualification.
- Preserve unowned WIP: `scripts/coreweave/prebuilt.py`, untracked
  `scripts/neuralese_data/check_vllm_lfm3_parity.py` and
  `scripts/vllm_fork_wheel.py`. Do not stage them under root's work.
- External Claude/Sonnet owns older69-worktree integration and is actively
  landing service-oracle/pipeline, source-custody and harness fixes. Today's
  bounded audit is
  `/home/werg/data/dgxops/claude-recovery-20261028/worktree-reconciliation-20261029/today-handoff.md`.
  Nine clean1029 trees are patch-equivalent to main; inspect live references
  before removal. Historical differing commits often have current equivalents;
  do not blindly cherry-pick based on ancestry alone.

## Storage and local job supervision

At13:46 internal had~53GiB free, external~95GiB. Goal floors: internal45,
critical32; external80. Measure before new large copies. Use bounded fsync and
cache release; do not hardlink mutable files or change live source link counts.
Retained142 uses12.5GB external; diagnostic33 uses3.46GB internal.

An obsolete48-minute `rg --files /home/werg/data /mnt/external -g mass-census-v4.json`
was saturating the HDD with directory reads while the exact census was already
known. Root verified PID2781071 and stopped it. Do not repeat broad disk searches.
The late local-validation `checkpoint.pending` is incomplete ZIP data (~16GB),
not a qualified checkpoint. The local Claude job was stopped as redundant and
timed out saving to HDD; do not call that an OOM or automatically restart it.

Re-check the ledger and systemd jobs, not this document's stale process snapshot.
Static Big2/render queues, admitted-source exports and historical publication
watchers may be useful. Do not stop other projects' workloads or conflate them
with Luna generation. All DGX tests/builds/probes use `scripts/memory_ledger.py`
with explicit Python/Node/PYTHONPATH and disposable hash cache.

Project Python: `/home/werg/natlang/.venv-neuralese/bin/python`.
CoreWeave SDK Python: `/home/werg/.local/share/natlang/cw-venv/bin/python`.
Node24 bin: `/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin`.
Use host `/usr/bin/python3` for pidfd ownership checks. Stop exact owned PIDs or
systemd units, never process-name patterns.

## Supervision transfer and final observed state

Root contacted active coordination identity `dgx-claude-5c7cf3fb` through
`scripts/coord.py --as dgx-codex-root`, request subject
“Owner requests convenient supervision handoff from Codex to Claude Code”.
Collect the final QAT/data handoffs and current process observations before
claiming the transfer complete. Child helpers are engineering strands, not
durable supervisors after root stops. Keep useful training/export services alive.

Detailed auxiliary handoffs:

- `/home/werg/data/dgxops/four-sft-cutover-20261029/OPERATIONS_HANDOFF.md`
- `/home/werg/data/dgxops/four-sft-cutover-20261029/data-readiness-handoff-20261029.json`,
  SHA `36441bd49d6859747135ce637b79a2973ad592dd0975b8232e3169608ba8a6f0`.

This section will be completed at the actual stopping point with live learner
steps, coordinator/replica evidence, recovery status, test status, commit/push and
the exact outstanding work. Earlier intermediate facts must not be mistaken for
that final observation.
