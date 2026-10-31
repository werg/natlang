# Supervision status, 2026-10-31 midday

Observation window: 10:43–11:29 CET. Recheck actual processes before acting.
Primary supervisor: Codex; active goal remains continued capable LFM3 native
Neuralese/Natlang training, not completion of this status document.

## Actual training and behavior

SD round8 completed, global step40, trained_rounds9. Full parent:
`/home/werg/data/lfm3-rl-dgx-20261029/sd-r1/round-000008-attempt-1/learner/checkpoint.pt`,
SHA `391a28d192939fb3e26ac8df8c968e2c1ab651cb1645010ea02c4d35de984ff3`.
The previously deferred text-result writer graphs were appended with full
optimizer/RNG/exposure preservation and the completed source epoch includes them.

Actual next allocation: `lfm3-rl-sd-expanded-13.service`, controller PID2037502
at observation, actor PID2068258, controller runtime `runtime-7b7b04444`.
It collects SD round9 from the earned step40 parent. Source adoption is real:
`round-000009-selection.json.source_sha256` equals
`618757192ad27cd73262ba308883915079016ee3f12bfccf3195c3af617810c2`,
54,881 available train cases. Both asset roots were supplied as actual unit
environment variables. The old state's `available_tasks` field is stale until
collection accounting commits; it is not evidence that adoption failed.
SGLang actor logs showed 12 concurrent requests at approximately620–680tokens/s,
varying with sequence lengths. Controller stdout's Transformers custom-code
prompt did not block actual serving; verify actor logs and sampled trajectories.
At11:16, 74/96 worker journal finishes existed. These are collection progress,
not a completed next learner update.

The step40 development probe reused this actual actor, with two client slots and
no second GPU model. Eight protected development cases, indices
0/16/32/48/64/80/96/112: 6successes, 2semantic mismatches, no invalid action,
repetition, budget exhaustion or infrastructure failure. Actor checkpoint bound
before and after. Evidence:
`/home/werg/data/lfm3-rl-dgx-20261029/eval-sd40-shared-dev8/budget-probe.json`.
Ticket classification and multi-hop ownership traversal failed. This is a
development baseline, not measured lift against a matched base-model run.

Critical checkpoint inspection: `native_full_sft` has **zero source graph visits**
and591,181 original obligations remaining. Foundation configuration did not make
RFT consume that stage. The source recovery helper is preparing an executable
earned-parent continuation with288 additional verified turns, exact objective
transition, original quotas preserved, and an early20-update behavior check.
Do not apply its earlier fresh-init candidate config. Actual current parent's
supervision objective hash is
`786931296e3ed658dc8e25ff6fcb33e38cb27fb6a262294588a268003cb31ae6`.
Root review rejected that first candidate: it used a sandbox-only base path,
mapped only the288newrows instead of all591469, and inherited older uncurated
sources instead of prepared policyv4. It is now explicitly non-launchable.
Terra owns the actual v4 build with currently ready scored supply and the288turns;
`lfm3-v4-curated-build-delay.timer` is verified armed for11:33CET after the bounded
recovery I/O window. No valid v4 mixture output exists yet. Its builder now retains
full-only/header-only teacher packages rather than silently dropping unmatched
prompt views (4486b8372;9targeted tests passed).
Joint SFT must become operational between RL windows promptly, retaining full
recipe obligations rather than treating a short window as stage completion.

JOLT round7 remains partially collected on its old bound runtime/cases, parent
round6/global42. Preserve those traces and exact bindings; no second actor runs
during the large SD learner updates. Cloud cw1..cw4 still lack GPU entitlement;
the autopilot's provider failures are not running training allocations.

## Source and generation fixes

All70 eligible temporal bAbI and338 multi-hop CLUTRR source rows now call their
actual named helpers, not erroneous `nl(facts,question)` / `nl(sentence)` calls.
All408 reference replays passed, with all declared wrong controls rejected.
Removed the first48 cap which left known failing worlds on the old framing.
Verified merged54881-row successor:
`/home/werg/data/broad-rl-tasks-20261031/assembled-guidance-helper-v2.ready.json`.
It is staged in `sd-r1/pending-switch.json` for the next **unbound** collection;
the currently bound SD9 collection retains its actual historical source.

The teacher launcher requested text emulation, but the frozen939runtime had
removed it and silently ignored unknown flags. A completed reviewed replacement
transport is committed as5c814d636. It carries actual teacher text, exact
producer/read edges and response receipts, without invented vectors/native state
or tokenizer claims. Local producer→consumer collector tests pass. Conversion
guards and Python corpus ingestion were committed79a544ce2 (16Python tests pass).
Fresh runtime `runtime-codebase-79a544ce2` and reviewed plan now have three actual
Luna/Pi workers in `luna-c-v5-deep-dispatch-79a544ce2.service`. Campaign is
`/home/werg/data/luna-deep-codebases-20261031/campaign-v5-c-deep-live-v3`, plan
3c19524d…3b92c5, runtime receipt38d92b56…6553a6b. Actual traffic is observed;
terminal results and SFT qualification are pending. Mock HTTP results remain tests.
This `/1` teacher freeze is not a `/2` sealed native-training runtime.

Recovered100design-brief applications, historical114Luna trajectories→288SFT
turns, separate180trainRL gates, and older semantic slates/feature curricula are
crosswalked in `PROGRAM_SUPPLY_AND_FEATURE_COVERAGE_20261031.md`. Do not equate
program ideas, executable task gates, positive demonstrations and actual exposure.

The7608Natlang-codebase verifier had OOMed during final output construction.
Final output now streams and resumed results bind input, executable closure and
Node binary/version. A rerun exposed a second launcher bug: systemd resolved bare
`node` using its manager PATH despite the admitted child PATH. Fixed launch
resolution in44e0e7464;70targeted tests pass and memory guard restarted on the fix.
`codex-natcode-node24-1031.service` now runs explicitNode24/frozen7b runtime;
initial rows genuinely pass (the earlier all-failing Node18 rerun is diagnostic).
It finished6059kept (5353train/305dev/401confirmation), zero wrong accepted, but
1549source failures and3673negative-control execution errors outside the kept
source. A confirmed the dominant cause: absent wrapper-root `natlang.json`
made prefixed `uses` paths resolve under the wrong package root. Fresh build-v4
includes that boundary and is being verified; build-v3 is preserved. Also inspect
109type-typo failures and distinguish signature/load-only repairs from actual
behavioral implementation validation before admitting the new task source.

## Recovery, I/O and next work

Discovered live `scripts/prune_rl.sh` deleted old full parents and exact objective
inputs before independently verified recovery, while trainer-owned async copy
children were killed when transient units exited. Disabled that destructive
pruner immediately; preserve dependencies until a complete recovery audit.
Do not infer a verified backup from `.offload.part` files.

Independent persistent copy:
`codex-earned-parent-backup-1031.service`, script
`/home/werg/data/lfm3-rl-dgx-20261029/ops/backup-earned-parents.py`.
Destination `/mnt/external/natlang-supervision-recovery-20261031`.
It copies and independently rereads full SD/JOLT optimizer/RNG/exposure parents
and observed controls. It explicitly does **not** claim complete source dependency
closure. At11:13 the SD partial reached1GiB; receipt is not yet committed.

HDD readers were starving recovery writes. The temporary watcher
`codex-backup-io-priority-1031.service` froze observed Natlang data/render/mining
units only, with an automatic20-minute thaw or earlier backup-process completion.
It logs exact observed units under `ops/backup-io-priority-20261031.jsonl` and
thaws on graceful termination. Recheck those handles; do not leave work frozen.
GPU collection, code fixes and local verification continue. Internal headroom
was roughly90GiB; recheck before large checkpoint/scratch writes.

Priority: finish actual earned joint-SFT continuation, finish honest Luna text
transport ingestion and launch the three real cases, publish the newly qualified
Natlang-codebase supply, verify independent recovery, restore all paused I/O work,
and keep useful GPU work continuous. Main was pushed through09c9cd700; later
commits need the current quick check and push. Full check remains unfinished under
resumable preemption; do not claim it passed.
