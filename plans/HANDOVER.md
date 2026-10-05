# Current handover — 2026-10-04

Corrected paired protected evaluationv3 completed: base1/23 successes vs final11/23,
final10semantic failures/2incomplete/0resource or infrastructure failures, plus one
held case. Small packet, not whole-system accuracy. Original published tokenizer's
apply_chat_template(tokenize=True) IDs exactly match training/corrected serving;
old serving had [BOS,BOS,...]. No retraining required for this formatting defect.
Post-training v2 now owns the GPU after paired eval; inspect its actual journal.
Pipeline builder accepts the explicitly labeled projection method, preserving
ordinary render/audit stages; candidates remain unpublished until reviewed.

## Student post-training and BOS correction — 2026-10-04 evening

Main350M SFT completed: global13164/105311examples, heldout2.26472→0.57536,
full optimizer/scheduler/RNG/adapter checkpoint saved. No merged export.
Frozen trainer and ready corpus correctly use one template BOS, no extra special
input tokens. Serving had a duplicate-BOS bug; corrected with a regression check.
Old periodic CPU evaluations and GPUfinalv1 remain historical, formatting-affected.
CPU watcher stopped and its owned13000 container stopped; retained partials.
Corrected paired GPU base/final eval under service
`natlang-final-student-gpu-eval-v3-20261004`, output
`runs/student-posttraining-20261004/final-gpu-eval-v3-single-bos`.
v2 failed at /select before task requests; v3 corrects URL construction.

Projection request-boundary MH implementation and completed-parent phase handoff
added; see STUDENT_REWRITE.md. v1 search interrupted/invalidated for generation vs
scoring BOS mismatch. New four-case v2 waits for corrected paired eval success
under `natlang-student-projection-mh-v2-after-eval-v3-20261004`.
Its64approved train references are all customer-service reducers; broaden before
training. No live post-training or new admitted corpus yet. Do not use heldout
case answers as training targets. Review successes as well as failures: six stable
successes include file edits/helper and all-false classifications; validate class
balance before claiming robust semantic skill. CPU resource failures are unscored,
not negatives. Actual outcomes and failure evidence are under execution-review-v2.

Targeted training-image checks:40passed/2skipped; new helper/tokenization checks
10passed (rerun after remaining edits). Host transformers is old and cannot run
LFM loss checks; use pinned training image. Visual pilot remains held; DGX jobs
belonging to the other developer remain untouched.

## Visual browser evaluator pilot — 2026-10-04

Source intake now has a real CPU-only isolated browser measurement path, not yet
native collector integration. Commit43a54fd5 adds pinned Playwright Docker image,
nonroot sandbox/seccomp, host-only scorer and two provider-free audit scripts.
See `scripts/visual-browser/README.md` and visual/frontend plan for commands and
limitations. Local image is sha256:14e2bdae6ea683940b3d495f904717d0537c29a9d5ca356e9ad7db458566b16d.
No source/oracle directory, key, GPU or host filesystem is mounted into the renderer.

Final pilot screenv7: all51 eligible WebSight packets checked,39measured,
12unsupported/unscored,7with baseline headroom,32already satisfy this objective.
All23mutation checks pass, including unchanged/whitespace equivalence and an
actual responsive repair. Early audit failures led to concealed-text, overlay,
pseudo-content and desktop-design guards. Generated pseudo-content/ambiguous
paint sources remain held. Screenv3 is superseded because a scorer rebuild
overlapped its run; screenv7 confirms an unchanged scorer hash.

Next work: broader paint/decorative-region verification; review Luna's topic
grouping/episode proposal; build native role-closed skill episodes and variants;
finalize reviewed admission; freeze a new
runtime; screen support and collect verified improvement trajectories. Do not
mark the734source packets executable or admit browser examples yet. Current
visual training admissions remain zero. Existing queues/runtimes untouched.
Central scorer dispatch and collector/screen code pins now include the pilot
objective; immutable image ID lives in host expected tasks. Training export
explicitly holds this pilot metric until paint/design and source-role review.

Additional review fixed empty zero-size links being treated as required visible
affordances. Five cases now have positive source-baseline headroom; two have
narrow-view gate failures. A host-authored narrow-only wellness CSS repair scored1
with all gates (wellness-repair-proof-v3.json), demonstrating repairability without
claiming a model trajectory. Placeholder restaurant text stays excluded.
Luna's initial/v2 role proposals missed six cross-role overlap edges. The new
`scripts/close_visual_source_roles.py` closes both declared whole clusters and
every initial overlap edge: v4 proposal has51tasks/13components/zero crossing
edges (support21/query16/transfer14). It remains review-only; all source IDs
are unassigned in the actual registry. v1/v2 proposals remain historical.
Native episodes and richer support diagnostics are next; proof/code/input hashes
and per-measurement hashes now support reproducible review. No model calls yet.
WebSight pilot progress is now registered in `training/self_improvement_tasks.json`.
The central inventory verifies report bindings and every saved measurement hash,
and reports51screened/39measured/12unscored/7source-headroom separately from native
targets and admissions. Re-running intake for the identical preparation retains
pilot progress; a different preparation cannot inherit it. Verified both paths.

Monitoring observed local main trainer at12770/13165 and DGX Qwen health200.
The supervisor status JSON has a stale checkpoint_step; current progress came
from actual trainer log. Git imported the other developer's new Neuralese work
from origin/main f34bf596 without conflicts; those experiments remain theirs.

## Origin integration and local cleanup — 2026-10-03 evening

Reviewed local changes committed as2a31eb4 (sparse-count/scene contracts and
existing contract tests) and6e8bd63 (full IR change receipts, existing failed-finish
handoff tests, generated-cache ignore). Three unreferenced superseded MuSiQue
prototypes preserved byte-for-byte under runs/local-state-cleanup-20261003 with
hash receipt; reviewed replacements remain tracked. Merged origin/main fe8c0b2
into local main without conflicts (merge b1f5e7d), preserving both histories.
Incoming Neuralese/skills work is explicitly in progress; no experiments launched
by this merge. Compile version5→6 requires rebuilding generated JS before using
new runtime. Existing production runtimes remain frozen. Focused read-only review,
merge preview and whitespace checks performed; no tests/builds run.

Fixed incoming full/sample data-build scripts' destructive same-tag reruns: no
rm-rf of prior final/closed output, validate single-component tags, unique default
tags, exclusive fresh run-directory reservation, per-run protected-source snapshot.
Full script now refuses an existing run, including interrupted output; resume a
retained partial run through explicitly selected CLI stages after reviewing it.
Do not restart the old whole-build script over the35GiB retained DGX partial run.

Monitoring follow-through remains active: evaluatorv6-v2 provider-free preflight
passed; corrected frozen server/bind and per-spawn pin checks await root launch.
Qwenv5 actual ARM native proof passed1024/4479, but three composites contain the
new exact ledger-source review candidate38f346b...; root source adjudication needed
before launch. Bunny/Luna v46 source packet prepared with zero provider-history
collisions and current native joins; no approvals/launch yet. Current Luna workers
continue, main trainer continues. Full rejection review under generation-check-
20261003/manual-rejection-audit-v9v4-v45-v3 distinguishes model errors from file
write and incomplete attempts, with one source/oracle question flagged.

## Monitoring resumed — 2026-10-03 evening

Latest actual full Muon checkpoint6120/13165, trained48960; fixed128 heldout
masked-token loss decreased from0.874923 at3500 to0.836124 at6000, zero skips.
Local GPU100%/7259MiB of8188MiB; main trainer and W&B active. Qwen v9-v4
finished/imported1024 (997admitted,27wrong_return); Bunnyv45 finished512 fully
accounted. Both need refill; v5 residual and cross-teacher refill preparation
assigned to existing Luna source agent, not launched. Luna v45 live228/219 of256.
CPU execution eval v4 failed19:55Berlin at server readiness (connection reset);
quality agent investigating before any restart. Broad development sync active
in push, initial full copy still incomplete. Local disk22GiB free.
Root monitoring state is active; resume50-minute sleep only after fixes/refills.

## Projection-sampling clarification — 2026-10-03

User explicitly superseded the rewrite-only-first interpretation: implement the
paper's iterative teacher-guided search, unguided student likelihood selection and
SFT, not its rewrite baseline. Existing collector remains unlaunched proposal and
verification infrastructure; draft v6 is not approval to launch the intended method.
See plans/STUDENT_REWRITE.md for required scorer/search, paper-versus-released-code
acceptance distinction, interactive replay requirements, and online tradeoffs.
Discuss online search/train rounds before choosing the live training cadence.
The historical section below describes the superseded implementation decision.

## Student-rewrite integration — 2026-10-03

User steered sleep into researching paper2610.02140 and aakaran's official
finetuning-with-sampling repo (pinned6d3e9f0bfaa98dcca534247dd35dc1b33dd8c428).
User then requested the simplest quick edge over ordinary SFT. Implemented cheap
student-conditioned rewrite collection and3-stage pipeline producer; details
`plans/STUDENT_REWRITE.md`. No MCMC, likelihood ranking, or live model/trainer
change.64 exact admitted teacher episodes passed frozenv41 preflight without
provider requests. Candidate plans remain unapproved; step5000 weights only
placeholder. Active evaluation services sealed on prior code stay unchanged.

Collector stores privileged wire prompts separately, records original logical
training contexts, executes full candidate episodes freshly, applies normal
contract/source/curriculum/native-link gates, emits native turns for normal
render/audit. Student server gains opt-in temperature/seed (default greedy).
Late-SFT launch, real yield/heldout measurement, inventory registration, and
reviewed Muon-preserving exclusion+append transition remain required before
publishing/training these candidates. No claimed performance improvement yet.

Latest observed main checkpoint5280; DGXv4 imported252/1024 and advancing;
Luna v45 automatically rolled both slots and has31/42 finishes; Bunnyv45running.
Fullcache hash audit passed18861 files/28977 entries0errors; broad local sync
service restarted, enabled/active. Startup partial-directory scan is slow
because it walks the entire mirror; safe optimization proposal pending.

## DGX dev harmonization and generation — 2026-10-03 15:20 UTC

User installed FFmpeg; `/usr/bin/ffmpeg`6.1.1 is verified. Dev fixes commit
496705c makes `build:node` stage `@natlang/node`, adds root build aliases,
uses root workspace setup, and honors NATLANG_DATASETS in acquisition/world
bridge/CommaQA checks. Bonsai test mocks health; CLI consent test isolates
config; source-review fixtures match exact pinned prompt/gold policy without
broadening holds. DGX independently acquired Neuralese commits: fast-forward
was refused, then a conflict-free merge preserved its history. Home has not
imported those independent features. Non-login SSH still defaults Node18;
set PATH="$HOME/.local/bin:$PATH" to select installed Node24. Build and48
application outputs succeeded;44 focused affected noncache checks pass.
TAP: DGX `~/natlang-remote/dev-harmonization-focused-20261003.tap`.

Priority dataset cache copy completed18861 files/4572455008 bytes. FOLIO,
TextWorld and CommaQA subtrees verified322 files/341 entries with zero errors;
cache symlink projected. All28 remaining cache-dependent reported checks pass
(TAP `runs/dgx-development-sync-20261003/focused-cache-tests-v1/tap.log`),
so the72 focused checks from this report now all pass. Full-tree hash audit
resumed on the slow HDD; broad enabled mirror remains paused until it passes
and the quality agent then restarts the existing serialized service. An unreadable
root-owned0600 training-readiness report blocked broad rsync; root changed
only that public report to0644, preserving it in the mirror.

DGX v9-v3 completed1024/imported1024 (1005 admitted/19 semantic rejects).
v9-v4 next1024 passed actual ARM proof/preflight and conditional gates;
`natlang-dgx-v9v4-reviewed-handoff-20261003.service` owns importer2559920,
remote residual-buffer1024 pool active and GPU observed96%.
Use immutable `pool-v9-alternates-v4/root-approved-conditional-controller-v2.json`
SHA edf93ac7e59da8ff998f8ab304ad2ab17b5d1265db77684f15ef7c47133d0989.
Original approved metadata omitted actual platform fields; root appended an
actual ARM/Node hash attestation in versioned native/assignment/authorization
v2 derivatives, retaining originals. A transient below12GiB preflight pause
was respected; unchanged resource gate passed later.

Bunny v44 finished512 exact accounted; old launcher/supervisor absent.
Bunny v45-r2 started512 under dedicated `natlang-bunny-v45-worker-20261003.service`,
launcher2560133/supervisor2560139, authority updated from actual status.
Approved plan SHA f8ac5eea1205af11434e8cbb97ea1c348229d3e87ffce1b22a72e82ff3c4359d.
Free Stealth only/no distillation flag. Luna v44 two slots continue; next
256+256 whole-group v45-r2 controller is armed under
`natlang-luna-v45-reviewed-slots-20261003.service`, approved plan SHA
 e5fe6c8a953d4571934539e90775971fd4f3f21c64eb6c7cc286d7b0ae4283b8.
It waits for each own bound parent PID exit and exact accounting; no extra
concurrent Luna slots. These are cross-teacher compositions, not fresh sources.
Local full Muon training active; latest observed checkpoint step4960.

New backlog reporter remains uncommitted/unadopted until independent-review
fixes are complete. Required: pin approved plans/receipts, latest attempt state,
actual artifact bytes+IR joins, queue identity joins. Preserve bounded reports;
there is still no trusted global distinct-target denominator.

## Target-count audit — 2026-10-03 14:20 UTC

DGX v9-v3 completed all 1024 and exact imports finished. Its preparation counted
4415 eligible compositions **before** selecting those1024, leaving3391 further
candidate compositions in that explicitly bounded universe. These reuse known
source questions; they do not establish fresh source coverage and require batch
review. At14:20 local queues had255 remaining provider attempts: Bunny40,
Luna1 116, Luna2 99. Counts advance; read actual journals before reporting again.

There is no trustworthy global distinct unfinished-source-task denominator:
`data/teacher/data-inventory/current.json` catalogs overlapping artifacts, not
canonical targets. Do not add the3391 and255 as unique source tasks or use
prepared training-row counts as generation counts. Snapshot:
`runs/monitor-cadence-20261003/target-backlog-1420.json`.
Luna quality agent is preparing a separate read-only target-status reporter and
registry proposal; root must review before adopting it. Source agent is preparing
the next reviewed DGX composition batch and identifying fresh source expansion.
The completed queue must not be restarted. Broad devmirror observed166GiB;
reload sync filters only after the first complete pass, as documented below.

## Source-copy filter repair — 2026-10-03 13:20 UTC

A broad `*.lock` exclusion incorrectly omitted 123 source `yarn.lock` dependency
files; the environment-file exclusion omitted one Git-tracked public deno-std
fixture. All 124 were copied additively and hash-verified, with receipts in
`runs/dgx-development-sync-20261003/source-filter-exceptions-*`. The sync plan
and helper now explicitly allow these reviewed paths before exclusion filters.
The active first broad pass loaded the previous helper/plan; restart its service
**after that pass completes** to activate recurring exceptions. Do not interrupt
the progressing first copy. Coordination locks and private credentials remain
excluded. At latest check the broad mirror was 133 GiB, DGX imported434 cases,
Bunny315 complete, Luna81/97 complete, trainer passed4370, CPUeval3500 at21/24.

## Current pipeline replication and handoffs — 2026-10-03 12:25 UTC

The user clarified that DGX development needs the entire non-obsolete pipeline,
not only final training outputs. Committed runtime, adapters, builders, training
and generation orchestration are carried through Git. The broad mirror retains
source corpora, unconverted inputs, intermediate IR, static outputs, generated
trajectories, partials/rejections, lineage, reviews, recipes, admission/split
proofs, and sealed runtime dependencies. Historical evidence required by current
lineage is retained conservatively; do not discard it simply because it is old.
Uncommitted prototypes are separately hash-preserved rather than silently applied.
See `docs/DGX_DEVELOPMENT.md` for ownership, paths, ARM tooling and containers.

At 12:24 UTC the broad external mirror was 106 GiB of roughly 260 GiB and still
copying. All 81 logical current-recipe inputs are already hash-verified. Do not
claim full replication until sync-status records the completed broad pass.
Recurring sync pulls `runs/dgx-development-generated/<unique-run>` home before
pushing the home-owned corpus; production DGX jobs retain the exact importer lane.

DGX v9-v2 finished with all 1024 artifacts imported. The reviewed v9-v3 successor
actually launched at 12:12 UTC and has begun successful imports (2 at latest
read). Current watcher is `natlang-dgx-v9v3-reviewed-handoff-v3-20261003.service`,
config `pool-v9-alternates-v3/root-approved-conditional-controller-v4.json`,
SHA b826a979d335a68d54e41b9502238f2a0f313a71738f404527569e0b04c3ff39.
Its owned importer PID is 2518616. Both Luna v44 256-case successor slots are now
running, PIDs 2518760 and 2512718; the current Luna controller remains v3.
Bunny remains under its dedicated v44 worker service. Earlier sections below
retain historical observations and superseded configs, not current authority.

Importer commit 47e1560 fixes premature terminal exit after a failed final copy:
finished jobs require successful transfer/import and exact assignment coverage.
Live rsync source-vanished events are recorded distinctly and retried. Temporary
write files are excluded, and rsync uses a 120-second I/O idle limit rather than
a wall-clock limit that killed progressing large transfers. Final v9-v2 coverage
and automatic v9-v3 launch were observed after this fix; no tests were run.

## Check-in cadence

The user requests an ongoing active-session loop: finish work, sleep 50 minutes, check training and generation, review failures and training progress, fix or restart what needs attention, then repeat. The current deadline is recorded in `runs/monitor-cadence-20261003/state.json`. This is an assistant sleep loop, not a background assistant scheduler. Routine agent progress notifications should not reset the deadline.

## Lifecycle fix and current authority — 2026-10-03 10:58 UTC

Root found a systemd ownership bug in the newly introduced transient handoff
services: start_new_session detaches a session but not the service cgroup. The
Bunny rollover exited and systemd stopped its just-launched child before any
worker status/journal/job evidence. Original launch record remains retained.
Recovery launched exactly one dedicated worker unit after proving old PID absent,
no worker evidence and empty log. An initial recovery command omitted two
binding fields after starting the unit; root reconciled that same live PID under
authority lock without another launch. Full receipt:
`refill-qwen-v9v2-cross-teacher-v44/bunny/root-lifecycle-recovery.json`.

**Current Bunny**: `natlang-bunny-v44-worker-20261003.service`, launcher2503351,
supervisor2503370, approved workerplan-v2 SHAe04d396d3095bb413852374f9057454fd2991607312ed8a586a6a77b464406d1.
Actual new journal3starts/2finishes atlatestread.512 customer-service cases on
v41. Future restart-only plan-v3 refreshes the incidental Luna wrapper import
pin; do not launch concurrently or rewrite current provenance.

**Luna successors**: two256-case v41 queues approved; root independently joined
all512 exact v9-v2 parent IR records and native256+256 cases. Decision counts are
1100+1078=2178 (agent prose incorrectly said1100each; artifacts correct).
Groups36/37 and sourceIDs553/570 disjoint. Same-provider saved17392 IR-bearing
rows/8001 exact signatures, no unreadables/collisions or active409 collisions.
Current waiting controller `natlang-luna-v44-reviewed-slots-v3-20261003.service`,
plan `luna-v2/root-approved-luna-plan-v3.json`
SHAfdd97040e7eb921bc77348884bafecfcd200952d776da58154f0b23f1c81b2f2,
record `root-luna-launch-v2.json`, waiting_for_slots/no children atlatestread.
Earlier waiting record preserved. Lifecycle-v2 controller failed before main
because root omitted sys import; corrected before any new worker or authority
mutation. Both original Luna workers remain active.

**DGX conditional successor**: current watcher
`natlang-dgx-v9v3-reviewed-handoff-v2-20261003.service`, config
`pool-v9-alternates-v3/root-approved-conditional-controller-v3.json`
SHAc22e461a362df500996d925c4db4d213d98809631b6e64f18deb6b8363f1f4ce.
Predecessor v9-v2 still active. Earlier watcher stopped before any v3 launch.
Root fixed all three handoff helpers to wait for owned child workers/importer
outside authority locks, rather than exiting and losing them. Commitcc6ce09.
Their controller-only512MiB cgroup caps were removed on the two current watchers
because these groups now include worker/importer children; existing per-case
resource limits remain. No blanket language deadline added.

Original misreported/failed lifecycle records, old immutable configs and journal
failure remain evidence. Never reuse an old armed watcher/config from the prior
sections. Current exact authority is in generation-check check.json and the
new lifecycle receipt; verify actual unit/PID/journal progress before declaring
workers alive. Training, CPUeval and broad devsync continued throughout.

## Bunny continuation and evaluation accounting — 2026-10-03 10:45 UTC

Bunny v43 ended with408 complete accounted keys plus reviewed incomplete index7;
never report409successful. The strict future rollover helper verifies exact
terminal event hashes, archives original failed events, and labels the parent
`finished_with_reviewed_failures`. Commit70d6062; helper SHA0d9a0aaec3db763feb7da0f9be65102c075a735a164d7299d70f76d99aeefaa2.
Root reviewed512 exact parentv9-v2 IR rows, currentx64v41 native512matched /
2178decisions /0unlinked and every worker/runtime dependency pin. Root failure
receipt preserves3-turn index7partial plus exact terminal event hash
bec0c30ef06e9315fdbaffdb09294b810e9d67d479d632a09c2a6ea87c39f9e2.

The512 successor launched at10:45UTC under
`runs/generation-continuity-20261003/refill-qwen-v9v2-cross-teacher-v44/bunny`,
rootrollover SHA bbcaaa9f9dcd365cf4432a4833134feee7b2583abec55caaf28425ae95e1dee5,
launcherPID2502129. Startup/provider checks were ongoing at latest read; confirm
worker-status and journal before calling provider generation active. Free-only,
concurrency1, no distillation flag. Its512 cases all customer-service; intentional
cross-teacher payload comparison, not new source coverage. Old drafts retained;
preparation-v2 corrects unknown family and actual v9-v2 source-closure receipt.

Future CPU evaluator harness `execution-eval-v5-candidate` is prepared, not
activated (manifestSHA09f108236ed6096d00314a2553c423bd588129f7eae5cce062971b3774f8b71e).
It tags the runner-owned case resource deadline and separates policy-held cases
from provider attempts. Original v4 results remain: projected historical audit
has1policyhold /23providerattempts /8success /8semantic /2contract /2incomplete /
3resource /0infra. Trust audit only as projected old classification; future
explicit signal-tagged records supply stronger direct evidence. MainGPU350M
snapshot3500, DGXGPU96%, full devmirror48GiB and continuing at latest read.
Luna successors remain in preparation before next sleep.

## Continuation gates and review — 2026-10-03 10:35 UTC

DGX v9-v3 second1024 batch now has root-approved assignment/auth/native artifacts
and passed actual ARM no-provider preflight (jobs_started false). Receipt
`pool-v9-alternates-v3/root-approval-artifacts-receipt.json` SHA98c8129393629fb16229743a40fb74dfccd8ae69e618b75023eccc8783e9623f.
Conditional handoff is ARMED as home user unit
`natlang-dgx-v9v3-reviewed-handoff-20261003.service`, config
`pool-v9-alternates-v3/root-approved-conditional-controller-v2.json`
SHA6e5855354c0ae8fb7afbf87ee6c537e69d30ff39ccf1d7620b6fb3e68edd0ee6.
It waits for all1024 v9-v2 exact exports/imported artifact hashes plus authority,
source/native pins, inactive predecessor, global user pool exclusion and actual
server model/resource checks. Current blocker is predecessor_still_running.
Never manually launch a duplicate while the watcher is armed or a claim exists.

Independent Luna review found a real reporting bug: systemd-run could succeed
and the following is-active probe timeout, leaving a claimed record incorrectly
reported launch_performed:false. Root fixed probe failures and interrupted
claims to unknown; retained claims prevent duplicate launch. Frozen controller
SHA28f792b048cb1fc65b05c33751410a5f638cda1e40c1ecd2df0799cf353ffcdf,
commit4e51faf. Earlier unarmed approval/config v1 remains evidence.

Bunny next512 packet is prepared under
`runs/generation-continuity-20261003/refill-qwen-v9v2-cross-teacher-v44/bunny`;
root verified512 exact parent IR rows, native512/2178decisions/0unlinked.
Sourcefamily is customer-service512; do not perpetuate draft unknown family
labels. Next approval waits for fresh reviewed failed-finish reconciliation of
v43 index7 and corrected future-only rollover support; active frozen pins stay
unchanged. Luna two next queues are in preparation. Last current terminal counts
Luna142/205+147/204 andBunny395/409 including one explicit incomplete.

CPU2200 task eval completed24:8success8semantic2contract5incomplete1infra under
original classifier. The apparent infra row and two incompletes all hit the
runner600s resource deadline; exact audit sidecar retained, future trusted abort
classification fix in progress. No endpoint outage evidence. One Qasper case was
policy-held before provider use. Do not turn resource/held data into semantic
DPO negatives. Step3000 CPU adapter snapshot is now evaluating; mainGPUtrainer
continues without interruption.

## Full pipeline replication scope — 2026-10-03 10:24 UTC

User reiterates that DGX development needs the full non-obsolete pipeline, not
only final training data. Committed runtime/adapters/case builders/admission,
training/evaluation tools and documentation are in the Git checkout. The broad
mirror covers original datasets, source archives, IR, static and model-generated
trajectories, rejected/partial evidence, review and lineage records, recipes,
prepared/rendered corpora, and sealed runtime dependency/toolchain evidence.
At 10:24 UTC the mirror held about42GiB; the roughly260GiB first pass was still
running. Do not claim the broad mirror complete from the81/81 priority receipt.
DGX `plans/neuralese/` is pre-existing untracked development work; preserve it.

Eight local uncommitted adapter/prototype/controller files are hash-verified on
DGX in `/home/werg/natlang/.development-source-snapshots/working-source-20261003/`
(linking a retained archive under `~/natlang-remote`). Their manifest and binary
patch record source HEAD90212f27b. They are preserved review work, not applied
changes or sealed production approvals. The same snapshot is retained under
`runs/dgx-development-sync-20261003/working-source-snapshot-20261003` locally.
Secrets and base-model caches are excluded; separate Python environments must
be rebuilt for ARM rather than copied as executable x64 environments.

## Latest generation continuation — 2026-10-03 10:06 UTC

DGX v8 completed512/512 and importer reconciled all512 exact artifacts. New reviewed v9-v2 is active on sealedARMv41: `natlang-qwen36-workflow-v9-alternates1024-20261003.service`, campaign `runs/dgx-qwen36-current-train-refresh-20261003/pool-v9-alternates-v2`; root importerPID2493985. Assignment SHAb2feb65d9db5e74574e0d4e2350e3f5d160fcc7753245dea27424807799b8b82, native review SHA0f256de00e4996a1922be5a00b82406b267f21da51de3bab07f13bd70a169c4b. Root independently checked3386 visible component question/state/gold joins and1024 exact native IR rows; actualARM proof4410decisions/0unlinked; full protected10338alias union disjoint. Helper rescanned6055history artifacts plus v7/v8 exact queued/imported payloads, no collisions/unreadable records. Actualremote no-provider preflight passed, server/model/global unit slot checked, predecessor full gate passed; root authority transitioned under lock and old512 assignment archived. `root-activation.json` records launch.

Preserve original native-review drafts: `materializer_accepted` incorrectly contained decision count4410 instead of case count1024. Corrected v2 draft and hashbound correction sidecar retain both quantities; physical compact IR digest and runtime canonical digest are distinct. New `scripts/approve_dgx_v9_pool.py` creates only new exact artifacts after root-approved binding and never launches itself. Failed check receipts remain. v9-v3 second1024 buffer remains pending root approval/conditionalhandoff. Controller and Bunny refill preparation continue before next50-minute sleep; no false promise of autonomous successor until armed.

Current dev recipe inputs verified81/81, full retained pipeline mirror running. CPU adapter eval2200 has preliminary8/22 complete successes, baseline1/24; two cases stillpending at latest read. Main350M Muon step3300+, zero skips. Luna2 andBunny remain active. Main training inputs remain fixed; chained append transition pending.

## Development replication and generation update — 2026-10-03 09:20 UTC

The user requests the full non-obsolete pipeline in their DGX home checkout, including source data and intermediate evidence, rather than only final SFT outputs. `/home/werg/natlang` on `ssh dgx` is built with separate Node24 ARM tooling and a Python development environment; `natlang doctor --json` passed against the existing Qwen8082 server. Setuptools package discovery is explicitly empty because this distribution supplies checkout script dependencies. Source updates use reviewed Git fast-forwards, preserving development edits.

Pipeline data is replicated into `/mnt/external/natlang-development-data` and projected through untracked branch links into real Git parent directories. Do not symlink entire tracked `data`/`runs` parents: that hides tracked files from Git. Root corrected its initial parent-link setup without data loss. Scope includes source corpora, adapters' input/output data, IR, generated/held/rejected/partial records, manifests, audits, rendered/prepared data and sealed runtime evidence; architecture-specific archived binaries are evidence, not ARM entrypoints. Base model caches and credentials are excluded. See `docs/DGX_DEVELOPMENT.md`.

At09:46UTC the priority data copy and exact81-input closure verification passed on DGX (zero errors); the broad retained-pipeline mirror is now running. Read-only rsync staging was corrected and the historical policy-code input explicitly materialized. The original recipe points to a mutable build path now changed. New replication manifest preserves all81 original hashes with one mapped archived code path; current dev code and original recipe remain unchanged. Enabled `natlang-dgx-development-priority-sync.service` waits for the active priority transfer, retries staged files if needed, and checks the pinned 81-input recipe closure hashes. Enabled `natlang-dgx-development-data-sync.service` starts the broader mirror afterward and repeats every five minutes. Plan/status/logs: `runs/dgx-development-sync-20261003/`. The passed `priority-verified.json` proves that closure, not the entire mirror. Root additionally confirmed prepared teacher8.45GB and rendered ready3.07GB visible in the dev checkout, return namespace linked to external storage, projection conflict count0, and clean Git fast-forward to9b09a1f. Home owns replicated production paths; DGX development results use `runs/dgx-development-generated/<unique-run>/` and are pulled home before each push. No deletes or in-place replacement; replaced revisions are retained outside the checkout.

DGX v8 is now active: `natlang-qwen36-workflow-v8-alternates512-20261003.service`, 512 reviewed alternate composites, root importer active, Qwen GPU observed96%. Packet `runs/dgx-qwen36-current-train-refresh-20261003/pool-v8-alternates-v3`. Assignment SHA5e3e25ae9970b00bb1d3c5ca43d81149783ff0135bbf21ef8aeb3cda18d6bf3f. Full protected heldout alias scan pins10338 combined aliases/792 recursive group aliases; zero selected overlap. These are alternate compositions of existing sources, not new source coverage. v9 1024-case successors and sealed ARM v41 are prepared but remain pending root approval/conditional handoff.

Index155 offline v8 reconstruction was independently reviewed and registered for future snapshot discovery at `runs/offline-teacher-replay-recoveries/index155-qwen-v41/recovered-v8.result.json` (SHA64e718a8cb5f333fe9d9a81b767e1fcf3438d98774fe38edef839fe1b93ec3c7). Root verified9/9 original full request hashes/context/tools, preserved model-response/raw-call fields, exact typed gold, zero protected alias overlap and retained offline trace using runtime localeCompare canonicalization. Zero new model calls; not a semantic DPO negative; active training inputs unchanged. Registration v2 helper failed closed on stale review field; v3 corrected helper used after root review. Original raw partial/error and prior diagnostic proposals remain. Do not execute old helper. Main training, W&B, CPU v4 execution eval, two Luna workers and Space Bunny continue. No cadence sleep deadline until current replication and reviewed successor work is complete.

## Latest sweep — 2026-10-03 08:25 UTC (ongoing)

Main source-clean-periodic-v3 Muon training and W&B observer remain active. W&B API independently confirmed optimizer step2860 and audited support counts15classes/108sourceIDs/196groups/4modalities. Dashboard https://wandb.ai/werg/natlang-training/runs/lfm25v13muon20261002. Audited support sidecar SHA79361498d9556e1e48155f1904f26df1ff90ab228b4ce198accdd2c3cd6b3994 is count-only; original unknown support fields remain preserved. Active trainer runtime/pins unchanged.

CPU execution-eval-v3 completed baseline24 with1success,13semantic failures,1contract failure,9incomplete after offline correction. One wrong answer containing “cancelled” was falsely tagged infrastructure by model-prose regex; corrected trusted status/checks classifier. V3 watcher then failed because signaling Docker CLI left its owned server occupying18084. Root explicitly stopped only that owned baseline container. New immutable v4 plan reuses exact saved baseline with hashbound correction, starts adapter2200, then500-step current-v3 snapshots; owned-label stop/wait/remove/port-free cleanup. Root verified40artifact pins and actual no-model preflight, approvedplanSHA6c643b788c32e155120e69ddd44225f67ba0d08c2fb7da2d90a734f8ef7a6290. **natlang-lfm25-periodic-execution-eval-v4.service active**, CPU container natlang-cpu-eval-v4-2200-adapter; no GPU/training interruption. V3 original artifacts preserved.

DGX409 finished409/409, all imports reconciled07:40; GPU collector currently idle pending source-safe512successor root review. Packet runs/dgx-qwen36-current-train-refresh-20261003/pool-v8-alternates-v3 passed actualARM native512/2175decisions and root1663exact question/state/gold/source-union joins. Earlier v1/v2 type-frontmatter proposals failed and remain diagnostic; onlyv3 eligible. Derived train-group closure is insufficient because selected components exceed r7 source set; full v13 heldout source/program/group alias scan required before approval/launch. Same teacher exact payload screening and final auth/native/assignment/launch/sync stillpending. Do not claim it is running yet. Luna205+204/Bunny409 remain supplied; Bunny index7 provider error incomplete preserved separately.

Sealed v40r2 actual ARM no-provider proof passed97focusedchecks plus real TextWorld completion vs fabricated-certificate rejection and million-binding preserved computation. Candidate not activegenerationruntime. Sealedv41x64 includes narrowly bounded provider finish error retries. Index155 nine actual Qwen saved responses replay to accepted result, current materializer passes; publication remains held because review receipt copied Bunny metadata (original_model in v1, saved_final_call still wrong in v2). Agent corrects independently derived v3 receipt and exact saved context/action provenance. Raw partial/failed attempt preserved; no new model calls or semantic DPOnegative. Future fresh Bunny7 retry must have separately identified new-runtime provenance.

Active sweep phase check_and_repair; next50-minute sleep deadline is set only once reviewed supply and repairs are complete. Production chained input appends remain pending; main inputs are fixed source-clean corpus.

## Current activation — 2026-10-03 06:33 UTC

Main training is now **`natlang-lfm25-full-v13-muon-source-clean-periodic-v3.service`**, container `natlang-lfm25-source-clean-periodic-v3`, run `runs/lfm25-350m-broad-20261002/full-v13-muon-source-clean-periodic-epoch1-v3`. Old v2 cleanly stopped at step2281/cursor18245/trained18248; every checkpoint file copied byte-identically with no shared inodes, including Muon/optimizer/scheduler/RNG. Receipt `runs/training-periodic-eval-20261003/root-exact-checkpoint-handoff.json` SHA3198a2b5a0d5f2da298a5492d81b396da16dcd191d3e984915527220b2ff0d34. Approved plan SHAcee0d3328f05fe1b64b4f57ef64a2f7551d59ad0f7569fc384850400067ffe0d; sealed runtime-v11 manifest SHA5e6224c429a9ecb59fb4b386529b4edd9934a9f2c21ca28bfd07ebbbf9754032. Existing source exclusions/protected split/target105311 retained. Old services remain disabled/operator-latched.

Periodic masked-token weighted heldout loss uses deterministic128 rows, initial resume observation then every500 optimizer steps. First observation at2281: **0.9176491843**,11935 supervised tokens,128 evaluated,0 skipped. Main advanced through2300; W&B server independently confirms2300 and this periodic metric. Dashboard https://wandb.ai/werg/natlang-training/runs/lfm25v13muon20261002. Separate CPU observer remains enabled; private key `/home/werg/.config/natlang/wandb.env` mode0600. Never expose key. Legacy2.2647 baseline is first100 example-mean loss and cannot be compared directly to the new metric. Periodic implementation commits d9204af/804db08, actual training-image8/8 CPU regression proof, including real Muon next-update/RNG equivalence. Full task execution evaluation is separately being prepared:24 protected nonworld cases, CPU-only server18084, immutable adapter2200; root approval required before starting model calls.

DGX213 ended with212 final exports, all imported, and index155 held with preserved RangeError/capture evidence. No live collector remained; a correct partial disposition is not full213 completion. Old sync2213713 stopped only after212 reconciliation. Source-complete **DGX409** successor launched06:33UTC: `natlang-qwen36-workflow-directory409-20261003-v7.service`, local `runs/dgx-qwen36-current-train-refresh-20261003/pool-v7-r7`, remote `/home/werg/natlang-remote/campaign-qwen36-current-train-refresh-20261003-v7/pool-v7-r7`. Same sealed v39 ARM/x64 and NVFP4 server,256 concurrent requests. Exact no-provider preflight passed; root independently verified409 embedded native IR/canonical provenance joins and1844 source question/state/gold/file dependency joins. Root-native-review SHA85a74e9750553d7a5720228bed88333683c4caef7f79765e4e26bc6e29fa3ab0. Assignment/activation receipts alongside packet; new syncPID2401448; authority updated under lock. Factory09d28ee rejects existing directory composites and missing state. These cases add task-shape coverage over existing sources, not novel source coverage. All world generation remains held until upgraded grading/runtime proof.

Native top-level IDs identify teacher trajectories; join via `task.program_ir.id` and runtime `recordDigest`, not that top-level ID or selection's physical JSON digest. Root corrected those review assertions before approval. Preserve held r3/r6 proposal and failed review attempts; no model calls were made for held packets. Runtime huge-binding capture and Story133 decisive-evidence metadata repairs remain in review; do not alter live frozen runtimes.

### 06:49 UTC follow-up

W&B API confirms main step2360; live checkpoint2340 independently passed later source-exclusion resume guard, cursor18717/trained18720/zero skips (`runs/monitor-cadence-20261003/periodic-v3-live-resume-review.json`). Periodic loss/counts are valid, but initial class/source support labels are all unknown due to missing flat fields in rendered rows; correct through a hashbound metadata join, never pretend unknown labels are meaningful coverage. Future source metadata correction is in preparation.

Bunny v42 finished669/669 (601 admitted/68 rejected), Luna v40 finished300+300 (287+298 admitted). Luna v42 runs71+70 successors. Root approved/launched Bunny **v43 409** source-complete typed directory cases at06:48UTC: `runs/generation-continuity-20261003/refill-workflow-v43/bunny`, rollover SHAeb1e45c8ee34ffad60f934c17851c8ac07da5ca00d4c7f8f7c11eb1c27e9f731, launcher2409733/supervisor2409734. Exact x64 native joins409/2253approved/0unlinked, identical reviewed DGX r7 payloads. Same-provider exact-payload history disjoint, but87/87 source groups have prior Bunny history; do not call this fresh source coverage. Automatic tools/free-only/no distillation flag/no wire seed preserved. Root archived draft prepared-worker-status, removed mutable status from immutable pins, supplied missing launch_record and approved worker-plan SHA before executing verified rollover. Further Luna supply in preparation.

Bounded scope display/capture and Story133 decisive-evidence fixes committed **0f9038b/6e0b86e/efc3764**, focused95/95 checks. Large bindings stay computationally intact; safe modest values retain paging; unknown structural equality conservatively emits changed-local notices. Incomplete durable representations have explicit diagnostic markers; materializer fails closed as unlinked unless same-turn raw model args prove the exact full action. No active frozen runtime changed. Future sealed runtime plus million-binding exact typedterminal replay, strengthened TextWorld grading/refreshed reference and ARM proof remain pending; preserve incomplete index155 raw evidence. Full CPU task evaluator remains preflight-only while watcher/current-output/packet-schema bugs are corrected; no CPU model server started yet.

### 07:06 UTC — evaluator and both Luna successors active; resume sleep cadence

Root approved/started **`natlang-lfm25-periodic-execution-eval-v3.service`** at07:01UTC. Packet `runs/training-periodic-eval-20261003/execution-eval-v3`; approved plan SHA53f667766e1c647ca4f2c1d3aea061b74330e37727983d946dacc8cf01b0981e, manifest SHAa2a05cbc7e67f339ff6d8e747764cf6e14b6b3e89c77c99ef4b675ef6bf96f1d. CPU-only6GiB/1CPU student server18084, no GPU flag; first real response200 at07:02.24 fixed protected nonworld tasks, initial historical2200 baseline+adapter then latest available500-step currentv3 snapshots with backlog coalescing. Score complete task outputs, contract failures separately; resource/infrastructure/ambiguous/incomplete remain separate and create no preference labels. No exact-action/code-string scoring. Root preflight fixed packet-schema mismatch, stopped-v2 watcher, missing copy-parent/log directory, firstbaseline coalescing; actual no-model setup proof passed. Two stale CommitPack gold digests were rebound to identical case IR with unchanged expected values; old packets preserved, root exact24-row review receipt. Execution outputs below `execution-results/execution-results/step-02200/`; logs beside mode directories. Main training reached2480 with zero skips and continued W&B delivery.

Root approved/started **Luna v43 205+204** at07:03UTC, plan `runs/generation-continuity-20261003/refill-workflow-v43/luna/root-approved-luna-plan.json` SHA35665f2ab0fa8847d935b1619a7cd4bc83e534fef4f7b424d14cd4ef5f8534ce, workerPID2417085/2417097. ExactIR/native/queue joins409; two slots share no source groups; all match reviewed r7 originals. Both v42 predecessors71/70 terminal verified. Root added missing supervisor/launch_record and exact helper pins; corrected cloned source-closure counts from actual6213IDs/191programs/790groups receipt. Do not perpetuate stale draft784/5957 labels. Same-provider payloads unseen, but all primitive sources/components/groups reused intentionally. Bunny v43 running25+finishes and DGX409 imported41 at07:04. Bunny case7 is held incomplete `Provider finish_reason: error` after3savedturns; raw partial/error retained for provider recovery, not semantic DPO negative. Other tasks continue.

Periodic support sidecar independently joins128 sampled IDs to the exact0dd7108 rendered corpus:15families,108sourceIDs/162links,196groups/379links. `runs/training-periodic-eval-20261003/periodic-support-audit-v1/joined/step-02281.support.json` SHA1aa97296ee2607789dfe9b452c709c77db84501994ed11a564d5e4aeda64c194. Original periodic record is unchanged; support fields became unknown because slim row index omitted metadata. Future seek-only hydration and explicit audited W&B support namespace in preparation; do not change activev11/v3pins. Future v40 million-binding typedgold/materializer replay and real-vs-literal TextWorld guard proof passed in agent work, but source manifest had stale provenance; new immutable r2 required before root deployment review. Never label unsealed proposals deployed.

Primary work in this sweep is complete. Resume requested50-minute active-session sleep; deadline in monitor state. Future runtime/support/transport proposals are pending for the next sweep; routine progress notifications do not reset the deadline. Current providers all have reviewed tasks; local Bonsai stays off.

## Cadence sweep — 2026-10-03 05:53 UTC (ongoing)

50-minute sleep completed; main source-clean-v2 Muon training reached2130, later2200. Active run remains unchanged while a checkpoint-safe periodic evaluation upgrade is prepared at the user's explicit request. Current legacy evaluator only evaluates first100 heldout rows before/after, example-mean loss; it is not periodic and does not score all5410 protected rows. Future periodic metric will use fixed deterministic128 heldout rows, masked-token weighted loss, every500 optimizer steps plus an initial observation on resume. Keep baseline/final labels separate; preserve weights/Muon/scheduler/RNG/cursor, frozen code packaging, source exclusion marker and protected split. Source audit agent implements future-only code and CPU production-path proof; do not activate unreviewed drafts.

User also requested W&B. Valid `api.wandb.ai` credential from DGX netrc authenticated accountwerg. Private local configuration is `/home/werg/.config/natlang/wandb.env` mode0600 (never print/read it into tool output). Separate CPU-only observer service `natlang-main-training-wandb.service` is enabled and active; code commit3b3afe9, SDK0.30.0 in isolated `.venv-wandb-report`, pinned `training/requirements-observability.txt`. Training container has no SDK and was not changed. Run `https://wandb.ai/werg/natlang-training/runs/lfm25v13muon20261002`, saved metric backfill and live telemetry. Server API independently confirmed optimizer_step2170 and actual numeric history records (`runs/lfm25-350m-broad-20261002/wandb-main-reporting-v1/server-confirmation.json`). Online local SDK queuing and remote confirmation are distinguished. Reporter survives resumes with stable runID; emits only scalar metrics/public config, not source examples/model weights; bounded512MiB/0.5CPU. Periodic completion JSONL hookup awaits upgraded trainer path.

Space Bunny256 completed and669 successor launched automatically, >528 completions by06:00. Luna2 completed300 and launched70 successor; Luna1 still238+/300,71 successor waiting. DGX213 imported203 by05:52 and205 raw results by05:59; seven remaining cases had fresh durable activity, so no blanket stall/timeout. Index155 has RangeError Invalid string length with partial evidence, not a semantic DPO negative; rejection agent investigates capture/terminal accounting. Semantic failures include typed boolean strings and relation/count errors; retain exact native boolean contracts.

Root found a serious proposed DGX supply bug: builder treated all4803 quality-v4 records as primitives, including existing directory-v4 composites; it copied only root question/state and omitted nested component folder files. Native literal-gold reference/admission replay could pass despite absent visible evidence. Exactly398/587 r3 composites are affected,189 not affected by that specific omission (`pool-v7-r3/root-visible-input-exclusion.json`). Whole r3/r6 proposal held; no provider launch occurred. Outer directory reducer inputs:{} is legal; the failure is missing component dependencies, not that root shape. Quality agent rebuilds from genuine state-bearing primitives only with independent visible-input/source/gold dependency-closure checks before ARM native proof. New aggregation of already-seen source evidence is useful task-shape coverage, never label as novel source coverage. Preserve diagnostic proofs/drafts; their native pass is insufficient quality evidence.

## Latest continuation — 2026-10-03 04:54 UTC (ongoing)

Main source-clean-v2 training advanced to step1830 with no overlength skips. Root independently validated the actual step1700 checkpoint for later resume beyond the exclusion boundary; weights/optimizer/RNG changed appropriately while the approved corpus/exclusion marker and cursor relationship remained valid. Receipt `runs/monitor-cadence-20261003/source-clean-live-resume-review.json`.

DGX56 finished56 exports (54 raw accepted/2 semantic rejects). DGX now runs213 source-reviewed nonworld cases: service `natlang-qwen36-current-train-refresh-20261003-v6.service`, local `runs/dgx-qwen36-current-train-refresh-20261003/pool-v6`, remote `/home/werg/natlang-remote/campaign-qwen36-current-train-refresh-20261003-v6`. Root assignmentSHA `03eead1aa8f6ca4eb2a90f0c296d98761b95f8f3ca7fa72b7b26a6042383c827`; exact no-provider preflight passed. Actual Node24 is explicitly selected by launch PATH, not merely present in the sealed runtime. 213 native gold/admission proofs are a transparent subset of actual227 ARM/Node24 executions. Frozen ARM444f5047/x6457042f53, concurrency256 (213 cases), existing NVFP4 server unchanged. GPU96%; syncPID2213713/importexit0. Fresh clean authority binding replaced the completed56 assignment under the shared lock.

Root found a real TextWorld false-positive grading bug: external world services were accepted by matching a certificate string without consulting trusted host completion state. Eleven iterate reference cases caught IterationLimitError and returned literal gold without world actions. Entire world scope is held from the new DGX pool until strengthened completion grading is deployed (14 legitimate quest cases also held as a precaution). Preserve raw records/golds and old proofs; do not treat these false positives as clean positives or semantic DPO negatives. Source audit agent is implementing isolated collector regressions and auditing existing main-run exposure; rejection agent repairs bounded TextWorld reference iteration separately. Active frozen runtimes remain unchanged.

Local Luna600 remains active (104/300 and138/300 at04:54); Space Bunny256 reached217/256. Quality agent prepares exact residual Luna141 and rederived Bunny669 with explicit payload/history/protected-source filters for conditional successor launch. Do not claim the rederived669 is the exact old residual member list, and do not label component-sharing curricula as novel standalone sources. Next50-minute deadline starts once these active repairs and handoffs are complete.

### 05:01 UTC follow-up

World guard committed `fc17cc4`; root independently reran both isolated regressions (2/2 passed). Existing same-source training data is a genuine Luna solve, with observed non-null certificate after real actions: no automatic hold. Pinned exposure report `runs/training-source-exclusion-20261003/textworld-certificate-training-exposure/exposure-audit-v1.json` SHA `a514965626e90220b0ec7b8ece514487832ce01b92d26e01d02fba7c047b6fd3` records step1860/cursor14877 and36 valid same-source rows (6 consumed/30 future), next20824. Agent's prose retained an earlier1840 snapshot; use receipt state, not stale prose. Guard is future code, not deployed in active frozenv39; all new world generation remains held until sealed upgraded runtime. Cached TextWorld IR lacked the bounded iteration already present in current source; refreshed IR/provenance checks are necessary in addition to fixing reference gold fallback.

Root armed conditional v42 successor handoffs: Luna71/70 planSHA `3ec05d3c20c66ab74fa5a3fdd3cb10264d61df31ca01dc0298659f5111ff92a6`, controllerPID2217903; Bunny669 rolloverSHA `e9b0786d4f2e67720725f1dea416bb418401c42e96cbb16ee45f16c3f8b10e1f`, controllerPID2217904. Paths under `runs/generation-continuity-20261003/refill-workflow-v42`; waiting on current300/300 and256 exact predecessor accounting. Root corrected cloned stale review counts/proof paths in separate approved copies, preserving drafts. Root verified810 exact source JSON/native/queue joins with zero same-teacher history/active payload overlap and no worlds. Source-row SHA in old proof is sorted compact UTF8 JSON, not physical line bytes; correction sidecar records both and verifies exact810 original line contents. Counts are payload-level freshness within existing source coverage. Future packet construction should create typed schemas from final queue/proof rather than clone predecessor metadata.

DGX first15 final artifacts imported by05:00 (lastpoll4 new; cumulative15). LFM training1850+ continues. Next supply agent prepares source-safe nonworld DGX successor, without provider calls/authority changes; refreshed world reference repair is independently running. Both work streams remain held for root review.

## Latest continuation — 2026-10-03 04:24 UTC (ongoing)

**Active training service is now `natlang-lfm25-full-v13-muon-source-clean-v2.service`**, enabled with reboot resume. Plan `runs/lfm25-350m-broad-20261002/full-v13-muon-source-clean-epoch1-v2/root-approved-training-plan-v1.json`, SHA `dc6c1c17bf90f5d1383d571d7842879beeb6eb3e334e60aa5e13d3be8aa1676c`. Runtime v10 manifestSHA `748425767ff3cc151414adda491f97c55bfcb9581eb098472203f6b9f11e9b60`; actual supervisor v3 SHA0b234c26 is separately frozen. Target105311 examples, original13165 scheduler horizon;105308 clean unique train rows,5410 heldout,31.5256% reducer share. This continues the main full epoch with Muon, not a pilot or optimizer reset. Old original and source-clean-v1 services are disabled/operator latched, checkpoints preserved. Never restart them as current training.

Root stopped original training cleanly at1686/cursor13488/historical13488. The reviewed exact exclusion manifest `runs/training-source-exclusion-20261003/tatqa-and-bugabula-v1/root-approved-transition-final-boundary-v2.json`, SHA `6f5563d346c8a2048342963a0fadcddc07759c17dbda00e03b2e0ecbcd0c52dd`, excludes8 TatQA source rows and1 Bugabula row. Three harmless reads were historical; six source rows were future. Rebased cursor13485, target105311. Original weights/184Muon optimizer/scheduler/RNG copied byte-identically without shared inodes; source data/golds untouched. Original checkpointSHA dfafcc1a214e57ff5c6051abfdb492ec385fcb7daa8181347a64fd3419aa740e.

Root discovered/fixed production issues before or during handoff: repeated parent-hash equality would break later resume; mutable source/code pins; incorrect Docker policy mount; obsolete supervisor missing `--expected-plan-sha256`; frozen-policy path rejected by original readiness gate. Narrow alias validation now preserves original ready/report hash/path identity while verifying sealed policy bytes; ordinary validator is unchanged. Production startup then exposed `for held in extra_holds` overwriting the heldout list (12 dictionary keys reported instead of5410 rows). Root stopped at1689/cursor13509/historical13512, preserving3 valid optimizer steps. Runtime v10 renames this local and checks heldout list/count/row IDs before loading the model. New disjoint v2 run copied every v1 checkpoint byte unchanged and retains the same approved exclusion marker/corpus identity. Small CPU Muon serialize/second-resume proof passed, but did not cover this full main-function variable collision; record that limitation. Production v2 startup/advance still being watched at this entry.

Both Luna workers now run300 new cases each at `runs/generation-continuity-20261003/refill-workflow-v40/{luna1,luna2}`, workerPIDs2170688/2170697; controller2170686. Root combinedplanSHA `d7db03dac97fe25b0807bc1d84b47a10e547901463d44d1d4936595192a453e5`, exact600 native/IR/history joins, full protected test closure. Bunny runs256 at `runs/generation-continuity-20261003/refill-workflow-v41/bunny`, workerlauncher2173704/supervisor2173705; rootworkerplanSHA `c39a7f0f883e5f26508f94f56289926393e8faaa26066c33052b32c4f2d81166`, rolloverSHA `6558d08e2da4d1acfc50996f01a3f2ec42cc44520735b51b827bfcb5663b1226`. Actualstatus is beside workerplan (`bunny/worker-status.json`). Old v40 Bunny bad draft preserved; v41 uses normalized native counts and path→SHA pins/current launcher73256bd. All selected tasks are **single-call workflow cases**, not new directory reducers; payload-fresh for each teacher, protected groups train. Cross-teacher trajectories are intentional; no novel primitive-source claim.

DGX56 finished **56 exact exports,54 raw accepted/2 rejected**, no error/retry/partial. DGX currently awaiting fresh reviewed successor. Review `runs/monitor-cadence-20261003/qwen20-dgx56-rejection-review`: Qwen20 rejected8 =5 semantic errors,2 model edit-string errors,1 quiesced/no-answer (not preference negative). Exact edit replays passed on x64 and actualARMv39; no edit-tool bug found. Two DGX56 failures are pending-cancellation resolution and a line adjustment exceeding explicit entitlement. Preserve raw rejects; no broad prompt/oracle relaxation. Rejection agent now prepares larger source-safe Qwen supply and may offer small explicit failed-only recovery meanwhile.

One-time preparer/apply commands ran from workspace scripts whose bytes matched frozen controller copies. Sealed trainer resolves `code-snapshot/scripts`; direct invocation of controller copies is not yet self-contained because runtime-root `scripts` is absent. Do not invoke those controller files directly until a future isolated package provides local import resolution. This does not affect the active trainer or reboot supervisor. `state.json` cadence is still check_and_repair; set next50-minute deadline only after active work completes.

## Prior cadence sweep — 2026-10-03 03:43 UTC

The fixed 50-minute deadline fired; the assistant is checking and repairing before the next sleep. Main full-epoch LFM training remains active: durable checkpoint1620/cursor12960 at03:53 UTC, original frozen corpus/plan unchanged. GPU99%/7331MiB at the wake. Three harmless TatQA read targets are now consumed; wrong calculation at position27429 and wrong final answer69963 remain ahead. Use the final quiesced cursor when preparing exclusions; never reuse older candidate counts.

Luna phase5 finished212/212 and211/211, **422 raw accepted /1 rejected**. Bunny649 finished649/649, **640 raw accepted /9 rejected**. Review `runs/generation-continuity-20261003/rejection-review-0343/review.{md,json}` found semantic predicate/relation errors and two answer-only format failures, no runtime/export/transport failures or justified broad admission relaxation. Fresh reviewed supply is preparing; do not reuse the164 directory-v4 tasks as fresh Luna work because all164 were already attempted/accepted. Composite tasks may share standalone component sources, but whole-task payload freshness and protected-test source/group closure are still required.

DGX20 recovery completed20 exact exports, **12 raw accepted /8 rejected**; sync finished. Its service is `natlang-qwen-repair20-v39-20261003.service`. DGX now runs56 source-reviewed WorkflowEvals directory-v4 cases under `natlang-qwen-workflow56-v39-20261003.service`, remote `/home/werg/natlang-remote/campaign-dgx-qwen36-workflow-directory-v4-supply-successor56-v1`; local `runs/dgx-qwen36-workflow-directory-v4-supply-20261003/successor-prep-v1`. Root authorizationSHAe60ccf4c025b46564b972c8cc024b8bc5182ed910f79ad0f68d35d4c16de4ac5, assignmentSHA4c1348b1167f61d79bb4ada9bbe69fc015e418fb8836398abab5feea8aa6c6b9, root launchplanSHA9ef3a1e704014f4eb0f7b488273ea7b08bc43b48992d72d71cdc33dbf9db8475. RuntimeARM444f5047 and x64 import57042f53 unchanged;56workers/concurrency256, executionplans true, transport8retries/5sec, existing high-throughput server unchanged. Actual ARM native56/56 proved;49 artifacts imported by03:55, no assignment holds. SyncPID2150832.

Root caught an earlier proof labeled ARM that actually ran on x86. Preserve its correction receipt and the actual DGX ARM proof separately. Native proof `.results.jsonl` contains case proof reports; use `native-reference-rows.jsonl` for actual native trajectories. Protected closure v2 scans all5432 test rows/nested identities (the original432 top-level source-only set was incomplete): exact selected56 programs/220sources/53groups have zero protected overlap. Superseded incorrect launch/settings/metadata packets remain preserved.

New exact MuSiQue source hold02ff4cc: `2hop__568389_161223` / `inline-curriculum:source_musique:a306e9c880f6a18c081c:v1` (Bugabula). The1400m gold comes from a bird elevation range, not the country's highest point. Preserve source/gold/raw; no invented correction or clean semantic DPO negative. Active training row `teacher-program:0a89f8bbbd02e9d7bed9:decision:0000` is at position39481, still future. Current v39 generation runtime contains the earlier TatQA hold only; its workflow-only56 queue is unaffected. Future broader runtime must include both holds.

Checkpoint-safe training exclusion is **not yet applied**. It removes9 exact rows (8 TatQA +1 Bugabula), preserves heldout assignment/original remaining shuffle, optimizer/scheduler/RNG/weights/history, rebases cursor only for already-consumed excluded rows, and retains original13165-step cosine horizon (one unused final tail step is intentional). Root blocked the v3 candidate: parent artifact equality on every resume would reject valid later checkpoints; mutable live source/code pins would break reboot resume after repository edits. Source agent is fixing both and preparing actual CPU Muon continue/save/second-resume proof plus an isolated runtime. Never apply stale moving-checkpoint candidates or mutate active bytes in place.

Commit73256bd fixes future OpenRouter terminal status: require explicit complete output accounting for every unique queued key, not merely exit0. Actual history independently classifies649/649 as finished and the old typed98 cohort96/98 as incomplete. New launcherSHA8b1078a1480cd114c12b67fafee12549e88e4bd7eb4c3b6a3b1b98b6aaa2e545; future plans must pin it. Existing raw status/journal history is preserved.

## Prior cadence sweep — 2026-10-03 01:43 UTC

Sweep follow-up at02:25 UTC: main full-epoch LFM run durable step1160/cursor9280, zero skips, finite losses; keep its frozen corpus/runtime unchanged. Both Luna phase4v2 slots finished64/64 and are awaiting reviewed replenishment.

**DGX generation is now running on the modern current-source ARM runtime** under `natlang-qwen36-phase5-crosscatalog-20261003.service`, campaign `/home/werg/natlang-remote/campaign-qwen36-phase5-crosscatalog-20261003`; local campaign `runs/generation-continuity-20261003/phase5-reviewed-supply/final-v3/qwen`. Fresh102 cases passed current native admission and full-v13 protected-source closure, zero provider calls during review. Runtime manifest444f5047eab29b76dc37e3c16f1c2948069308143a953cc36d4ed58ff4ec97d7; local import x64 manifest57042f535d255b7bed64fa96d288dcd55aa8d29f49ee340db66693425b67572e. Root plan6d6a5319fbe39b322edf81e711fc4f24fddae5e9c1759420cb99cced5649f2c9; assignmentceb65b33b35456b9d95cc97ba4082a1ee1063f2c69f597e5977e16c684bfafec. Case-event journal and final export behavior are now supported/proven. Systemd KillMode=mixed/TimeoutStopSec600 allows supervisor drain. Latest sync accounted81 artifacts at02:24 UTC; GPU96%. SyncPID2075007. Old951 finished951/951, raw924accepted/27rejected; import951 is assignment eligibility, not positive SFT admission. Historical717 request-budget failure remains preserved for separately reviewed recovery.

**Space Bunny typed98 is active** at `runs/generation-continuity-20261003/phase5-reviewed-supply/bunny/typed-return-v2-handoff`, launcher2074283/supervisor2074289, single-request concurrency. Actual status is beside the worker plan (`worker-status.json`), not under worker-1. Root planSHAe206915531a3f1408fb8a97a97d3990015d5f083623dd2dd0c6cab57cbc0c9e6. Source-visible field types were made explicit without changing gold/files or requiring delegation. Review-v4 SHA b70ed0aa72ae32768d040fbf7af41b46cfee49e66961447ba8d73c1816d6d5a3; native-proof-v3 SHA c8c55f3b68555dbdd3a5254bc74b818e09b45f93c2a4d7ca49f845a8bb61d353. First6 outputs all accepted; latest41 finished by02:25 UTC. Preserve raw older48 rejects;40 were schema ambiguities, not clean semantic DPO negatives.

Root caught mislabeled worker pins, rollover dispatching itself instead of the worker, and an incorrect status path. Failed wrapper exited argparse before provider calls; original one-shot receipt remains preserved. A separately recorded direct worker recovery started the approved unstarted queue once. Do not rerun the old claimed rollover. Authority now has a clean actual98-case binding; inherited predecessor proofs/counts archived separately. Generic preclaim launcher/path/status/pin validation and clean authority construction are being reviewed. A649-case cross-teacher Bunny successor and a Qwen historical-rejection recovery cohort are in preparation; neither is yet authorized to run.

New TatQA source hold6d7d277 (cross-quarter reference error) must be included in all future runtime/recipe admission. Actual training exposure receipt `runs/monitor-cadence-20261003/tatqa-eps-training-exposure.json` shows only a harmless table-read target consumed; erroneous calculation/final answer remain ahead. Explicit source-exclusion transition preserving optimizer/RNG/step is still pending; do not mutate the active frozen data. Append/protected TEST tooling d68321a includes nativeMuon CPU resume proof. Production chained append, exclusion transition and native TEST prompt-fit preflight remain outstanding; no corpus expansion deployed.

### Sweep completion — 02:53 UTC

Full LFM checkpoint1300/cursor10400, zero skips. Training-exclusion primitives/tests committedb217d7a; candidate binds stale1240checkpoint and is not applied. Current-policy readiness gate/new immutable transition plan still needed; active training unchanged.

Luna refilled423 workflow tasks (212/211), root planSHA d70751f19734ebf3ccaecd57d4d48694ea8097fe930516b0285af2261e6bb999, PIDs2084701/2084709, campaign `runs/generation-continuity-20261003/luna-phase5-workflow-successors-v39`; both producing results. Typed workflow adapter committed0aa9961.

Bunny typed98 ended96exports plus2 connection failures66/93. Original98 evidence remains incomplete; a copied-partial two-case repair at `.../bunny/typed-return-v2-handoff/transport-repair-66-93-v2` completed2/2; rootplan83fbd963ce9337424340e81d139ac710fd03526505f6a5838446502786832cb9. The649 successor is now running at `.../bunny-dgx951-cross-teacher-v3`; actualworker launcher2092139, rootworkerplanf9c5c69fda850b47c89a90eee1c6d377f4bd40e5a6d513eda0edcdfb465fe51e, rollover-v4SHA1904c2b68dbc9887fe46fb82cadc2cbe072f4381a90cc028966856cf2e08fec0. Earlierv3waiting guard failed honestly on incomplete98; preserve receipt, do not rerun it. Generic worker lifecycle still calls exit0 `finished` despite incomplete accounting; guard caught it. Future runtime connection retry fixfdcffd7 recognizes exact flattened Connection error message; active frozen runtimes unchanged.

DGX102 finished78rawaccepted/24rejected. Review `final-v3/qwen/rejection-review-v2`:7 label-correct zero-histogram contract mismatches,5SCONE state errors,5TatQA,6MuSiQue,1CLEVR. Future SCONE root rules clarified5fc9ed6. Source ambiguities remain review-only, no blanket acceptance/DPO labels.

DGX20 historical recovery is running under `natlang-qwen-repair20-v39-20261003.service`, remote campaign `/home/werg/natlang-remote/campaign-qwen-repair-20-v39-20261003`. Local campaign name is `.../qwen-repair-23-v3` but exact count is20 (prior drafts23 retained). Full-v13 exact program/source-ID/group test closure passed; ARM/x64native20/20, no current holds. Rootauthorization38ddaafa68bbb42cd741ad8474690b868b38b163310a6ff7400fbc0b825fdbf4, assignment9175e4c330883c9bfa57d54bf19294b84418acce025f4cc75e14b54b19cbe50a, planffd1ae0bfdcfeef17c108039933e9b66e0baffb5531427579fbc4908c4f8abb8; sync2095781. Root removed inherited old102 metadata/ARM importer from draft. Added missing local bundle chat-request-config after first importer ENOENT; subsequent import passed. Old102 raw results preserved; no overlap/accepted-task negative relabeling.

Next: review larger modern directory-v4 DGX supply being prepared by Luna quality agent; review checkpoint-safe exclusion transition/readiness gate; source oracle agent preparing it without pausing live training. Review new batches and accurate worker completion before handoffs. Distinct reducer/composite task coverage may reuse questions already answered standalone; component-level positive suppression is not a blanket rule.

## Original training phase (superseded by source-clean-v2 above)

The main full-corpus LFM2.5-350M run was active under the enabled user service `natlang-lfm25-full-v13-muon-epoch1.service`. Its immutable plan is `runs/lfm25-350m-broad-20261002/full-v13-muon-epoch1/training-plan-v3.json` (SHA `fcae226e198429c82134212f19e6a89d7a84a806bf58df0beb94dfe97d770733`). It trains rank-16 LoRA on a BF16 base, using Muon for all 184 trainable tensors. This is a full-corpus epoch, not full-weight finetuning. There are 105,317 train examples and 5,410 heldout examples; target 13,165 optimizer steps. Early throughput suggests roughly 44 hours, subject to sequence lengths.

The first full checkpoint was independently loaded on CPU: all 184 Muon momentum buffers, scheduler step 20, RNG, weights and data cursor were present. Baseline heldout loss was 2.26472; step-20 loss was 1.80621. The first 50-minute check found live step 370 and durable step 360 (2,880 examples), all finite losses and zero skips; recent step time 11.7 seconds implies about 41.6 hours remaining, subject to sequence lengths. Read `trainer-output/checkpoint/state.json` for actual progress; `supervisor-v3-status.json` retains a stale startup `checkpoint_step:null` until a lifecycle transition. Do not mistake it for a missing checkpoint. Commit be20d7d makes the repository supervisor `status PLAN` command report fresh validated checkpoint progress and supervisor liveness without rewriting the lifecycle file; four focused tests pass. The active private supervisor is unchanged.

The previous 500-step pilot stopped gracefully at step 351 and is preserved. The full run starts from the base because two pilot training groups occur in full-v13 heldout. Model, trainer code and inventory policy are physically frozen; future repository edits must not alter this active run. The supervisor retries trainer failures with backoff and the enabled service resumes after reboot. Its service has `Restart=no`; a killed supervisor or orphan container requires inspection before restart. Operator stop uses the private supervisor's persistent stop latch. SIGTERM checkpoints at a complete optimizer-step boundary; power loss recovers the latest durable periodic checkpoint.

## Current generation

The first 50-minute check found DGX Qwen1024 terminal incomplete: 1,023 exact exports and case717 exhausted its 384-request budget. Its status, partial, error and outputs are preserved. The original v38 gate correctly blocked. Recovery then exposed an original launch configuration bug (versioned journal basenames) and a new guard's self-lock check; both failed before model calls and their receipts are preserved.

**DGX generation is restored** under transient user service `natlang-bonsai-qwen-successor-gate-v40.service`, with the same approved 951-case IR and ARM64 runtime38. Gate r2 SHA123e8afb929428363419d52e0517cb868d8470a7c7cdbda3fe6abd773e99f9c1; pool authorization SHA3963d77bf2da69d8235de87eabbddcd7aa773ec5278510e44c20f34ee3366e16; gate authorization SHA90e1d4dd148ed1b51bd6087829965b8b0b961c56fd26eae02d3f77f4d91e69da. The actual runner preflight and the guard main path under its shared lock both passed without model calls before launch. Latest observation at 00:42 UTC: 535/951 results, zero error files; local training GPU 100%. Canonical `jobs/`, `worker-status.json`, `results.jsonl` paths now match the importer; control is `control-v40/`. Local `assignment.json` uses the frozen x64 current598 import runtime with actual ARM38 teacher provenance kept distinct. Sync PID2022153; latest importer accounted for 520 unique artifacts, with the last 20 all admitted and no holds. ARM38 silently ignores case-events-file, final-export-only and drain-file: there is no original case-event journal, exports merge progressively, and graceful drain is unsupported. Do not interrupt this healthy queue to add observability. Derived job inventories are labeled as derived; never fabricate original events. Use the current queue accounting to assess completion; inspect supported shutdown behavior before stopping. Old v38 gate is disabled. This v40 service is transient; after reboot or interruption inspect existing jobs/plan/fence and resume the same pool explicitly, never rerun the one-shot gate blindly.

Local Bonsai remains disabled. Luna workflow9/workflow8 and Bunny395 all finished; Luna7 successor finished5rawpasses2rejects. Phase3 native-reviewed supply is running: Luna1 30 and Luna2 29 (20 finishes each at 00:42 UTC), Bunny42 already finished. Phase4v1 blocked on protected-source intersections; v2 excluded 9 unsafe candidates from the shared pool, then passed 192/192 native replay and full-v13 selected source closure. Root approved and armed v2 at 00:53 UTC: Luna plan782fca052b178100974547ae04fc64fed66d77430000d743d05615b17f2d892c, launcher2037368; slot2 active PID2037375, slot1 waits behind phase3. Bunny rollovera1b01e29d273767eb8ad30ab8c8fbb5db3ac5db727cf33ba7233c9dc86c68252, launcher2037369; worker launcher2037374/supervisor2037402 active. Each has64 directory reducers. Keep v1 evidence. Source revisions shared across a dataset are provenance, not split aliases. New reducer compositions may use already covered TRAIN sources when the task shape is meaningfully different, with truthful parent lineage and source closure.

Cross-teacher source-group overlap is allowed for deliberate coverage and distinct task shapes. Keep truthful parent source IDs/groups, deduplicate redundant targets, and protect the train/heldout source closure. It is not a reason to invent a global freshness restriction.

## Outstanding work

Growing-data intake is requested but is not yet supported by the active frozen trainer. A design and CPU validator are staged in `scripts/prepare_training_append_intake.py` and the epoch directory. An append-aware consumer is now staged but uncommitted in the workspace, with11CPU tests passed; root review and exact optimizer/RNG/schedule transition proof are still pending. It preserves the old shuffled prefix and extends LR continuously at an explicit checkpoint. The protected-test packet v4 preserves 562 exact IR variants under composite identities; actual rendered TRAIN alias closure passed. Nineteen serialized IRs exceed 16,384 tokens; assembled native prompt/context preflight remains required. Append CPU regression uses AdamW; actual Muon append resume and chained append production manifests remain unproven/unsupported. Do not edit the active dataset or silently bypass corpus identity checks. Prepare new admitted batches as immutable candidates meanwhile.

The full DGX951 backlog contained only seven eligible Luna cases after same-teacher accepted/active suppression; all seven were native-reviewed and launched. Do not pad with repeated positives. The rejection agent is preparing additional meaningful recovery/new source tasks. Four legacy Luna2 TatQA mismatches involve representation or scale; re-admit against current reviewed contracts before labeling them true negatives. Numerical computational correctness must remain exact. Preserve rejected data and migration evidence.

---

The following entries are historical; newer controlling instructions above take precedence.

# Training efficiency audit — 2026-10-02, latest controlling work

**Main training is now running:** Enabled userunit `natlang-lfm25-full-v13-muon-epoch1.service`, plan `runs/lfm25-350m-broad-20261002/full-v13-muon-epoch1/training-plan-v3.json` SHA fcae226e198429c82134212f19e6a89d7a84a806bf58df0beb94dfe97d770733. Rootverified26artifactpins/image; exactcontainerpreflight audit/mix/inventorypassed. Physicalclosedtraincode/privatepolicyROoverlay, writableTritoncache, frozen supervisorv3 expectedplanhash. Fullcorpus105317train/5410heldout,13165stepsoneepoch; BF16base withrank16LoRA (5996544trainable),184Muon tensors/0AdamWaux, freshbase becauseearlierpilottrainedon2fullheldoutgroups. Baseline2.2647225933;step10finite1.9029508,~12.3sec/step,100%GPU. First20checkpointpending. Prior500stepAdamWpilot cleanlypausedstep351/2808examples,checkpointpreserved. Maininitialthroughputimplies~45h/epoch provisional,previous35h estimatecamefromsubset. Userunitenabled+Linger; crashretry5/backoff30-600s, checkpointalloptimizer/scheduler/RNG/cursor+fsync. Systemdservice itselfhasRestart=no; wrapperretriestrainerfailures, rebootautostart works; deliberatepersistentpauseusesprivateSupervisor stopPLAN. Suddenkillofsupervisor mayrequirecheck-in restart/orphanreview.

**Generation refills:** Luna1recovery10 rootapprovedfc1e0622...f2230,running1965299;Bunny128rootapproved3261d37b...72c361,runninglauncher1965727/supervisor1965763,current598execution/nativereview128/128/260decisions. Luna2old306completed;freshworkflow8 rootapproveda584339c...aebe handoffpendingactualPIDverification. Alloriginaldrafts retained; rootcaughtBunnyworker/rolloverplanconflation beforeclaim andfixedseparatecontracts; no providercalls oninvaliddraft. Cross-teacher sharedsourcegroups are not inherentlyunsafe: permitdeliberatecase/questioncoverage withtruthfuloverlapmetadata, same-teacherpositive/active-IDdedup andfixedtrain/heldoutsourceclosure. Do not imposeunrequestedglobalcross-teacherfreshnessrestriction.


**Controlling check-in cadence:** User explicitly requests an ongoing active-session loop: finish necessary work, sleep50minutes, then check training/data generation, reviewrejections/trainingprogress, makeadvisablefixes and restart/refillwhatneedsit, then sleepanother50minutes. FinishcurrentfullLFM/Muonautoresumehandoff andLuna/Bunnyrefills beforefirstsleep. This is an active-session sleep convention, not a scheduled assistant wake or an hourlybackgroundmonitor claim. Preserve the absolute wake deadline if subagent messages interrupt the sleep, unless they require work.

**Latest live check:** DGX1024campaign has883case finishes834rawpasses49rawrejects (latest22:06:58UTC),GPU96%; rawpasses stillrequireadmission. LocalGPU100% training; fullLFMrender emitted111301rows,tokenaudit active. Luna2repaircase290active; priorLuna1/BunnyPIDsended, rejectionagent reviewingnormalcompletion vsfailure andrefillingreviewedqueues. FullMuonphase muststartcleanbase: sourceoverlapcheckfound2previouspilot TRAINgroups in v13TEST, so reusingpilot-derivedadapters wouldcontaminatefullheldout. Preservepilotweights/results; do notalterprotectedtestsplitjusttoreuseadapters.

**Latest optimizer/resume requirement:** User prefers Muon and bgkit-style emergency checkpoint plus automatic resume, preserving all optimizer state. Root added `--optimizer muon`: nativeTorchMuon fortrainable hidden2D matrices, auxiliaryAdamW forvocabulary modules/vectors/3Dstackedexperts, match_rms_adamw LR adjustment, completechildoptimizer states and strictnamedshape/dtypepartition onresume. Newfullcorpusphase startsfreshMuonstate fromthepinnedbase becausepilottrain overlapsfullheldout; neverreinterpretAdamWstate asMuon. CPUoptimizer/checkpoint/loss regressions30passed/2CUDA-skips inactualtrainingimage. SIGTERM/Ctrl-C trainerfinishescurrentoptimizerstep andcheckpointweights/optimizer/scheduler/RNG/datacursor; SIGKILL/powerloss recoverlastcompletedperiodiccheckpoint only. Sourceagent implementingdurablesupervisor andtests; automaticreboot/resume is notyetdeployed.

**Latest training scope correction:** The user requests full training rather than an arbitrary500-step pilot. Current healthy local run `runs/lfm25-350m-broad-20261002/train-500steps-broad-v2` uses7993train decisions;500steps withaccum8 covers4000examples,only~0.5epoch ofthat subset. It is NOT full-corpus training. Preserve its checkpoint/optimizer/receipts and keepGPUbusy while preparing LFM-specific rendering/audit of the completed v13 model-neutral fullcorpus (105869train/5432test beforemodel-specific exclusions). Next phase must cover at least one complete audited train epoch, with periodic recoverable checkpoints and heldout-based further-epoch/model selection. Do not mutate oldrunidentity or scheduler to disguise an extension. Full v13 data-only DAG completed; helpercommit9580fc1,284tests passed/2CUDA skips.

**Controlling utilization priority (latest user steering): maximize useful GPU work on both machines.** Local RTX4060 is now training real LFM350M continuation `runs/lfm25-350m-scoped-20261002/train-continuation-100steps-v2`, from the first step100 adapter with a fresh optimizer/new run identity; latest root GPU sample **100%, 6715MiB**; durable continuation checkpoint is step60/100,480examples,zero skips. DGX simultaneously measured96% GPU utilization generating with Qwen. Broad LFM corpus is audited and ready:7993train/890heldout,seven over-budget rows held without truncation; begin500-step training at a safe checkpoint handoff. First continuation failed before model load because the wrong cache was mounted; preserve that failure. The pinned persistent LFM cache is **`models/hf`**, not `~/.cache/huggingface`; offline load now works without redownloading. Broader LFM selection/render/audit runs on CPU in parallel, then checkpoint-boundary handoff to the broader experiment. Do not interrupt useful training for further Ling microbenchmarks. Agent's “148 shards” loading report refers to loading tensor entries, not 148 weight files.

DGX live21:16UTC **96% GPU**,232running/1waitingrequests; Qwen1024campaign791exactexports233partials0errors. KeepQwengenrunning whilelocalstudenttrains. LingBF16download completed21:11UTC authenticated and**all32LFSweight hashes verified**,15,787,992,416bytes. Separatepreparedmodel nowat `/mnt/external/hf-cache/natlang-prepared/ling3-tiny-training-20261002`,32weightslinkedwithoutanother15GBcopy, exactpatchedsource65a6683f…14dd6. Localreceipts `runs/ling-completion-projection-check-20261002/dgx-{download-ready,preparation-receipt}.json`. NoLingDGXGPUload/training yet.

Bunnynew256queueactive launcher1924617/collector1924621:98MuSiQue70SciFact84modernTATQA4recovered; source-disjoint/current598sidecar256/256/1032decisions,zero holds. Actualexecutionusesfixedv38x64e16ded…afc86,distinctfrom598current-policy proof; preservehealthyworker. GenericOpenRouterhandofflog-parentpreflightfixed/tested2/2,commiteee8282; setupfailedclaimbeforemodelcallsarchived. Luna1reviewed64active,Luna2active,Bonsaidisabled. FurtherLingfull-widthPEFTprototypecompletedoneAdamWstepmatchingparameters/state buttestincorrectlydemandednonzeroLoRA-A gradientonstep1withzeroB; archivedfailure, correctedassertionpendingnextGPUwindow.

**Prior implementation rediscovered and CPU-audited:** `models/candidates/ling3-tiny-grouped/modeling_bailing_moe_v3.py` exactly matches committed `scripts/ling_grouped_experts.py` converter output for the pinned-compatible HF source; config/tokenizer hashes match. It provides stacked experts, conditional Unsloth grouped dispatch, checkpointed token chunks and custom512-position checkpointed causal loss (**not Liger**, correcting the initial inspection guess), with15GBconvertedweights. Historical RTX4060speed1177tokens/sec explicitly omittedrouted-expertLoRA; grouped_mm1158slowerthanUnslothTritonthere. Noadapter/runreceiptinmodeldir. Currenttrainerbatch>1requestslogitswithoutlabels, bypassingthemodel'sblockwise loss; anybatchedfixmustpreserveper-decisionmeanweighting. Reusepriorimplementationafterbackend/optimizer/adaptercoveragechecks. Audit `runs/ling-student-compatibility-20261002/preexisting-candidate-audit/review.md`,SHA1ded62f6e2f47a00d83ef669ed2598ae9e429bab6b7527759e7acefef2516f34. Do not mutate/remove model folders or benchmark on the occupiedGPU.

**Latest completed fixes:** `32c87f4` adds `scripts/ling_training_compat.py`, a separate model-directory installer pinned to the reviewed upstream model/config/configuration source hashes. It preserves cached originals, links weights instead of copying them, fixes Transformers 5.5 RoPE/FX compatibility, short-training chunk selection and integer completion projection. No sorted-dispatch or padding patch enabled. Seven unit tests pass. Actual reduced Ling + PEFT **CPU and BF16 CUDA KDA+MLA+MoE** completion-loss tests pass: 65 full versus 8 projected positions, exact matching loss and adapter gradients (max difference 0); original forward reproduces 65-versus-8 label mismatch. Actual CUDA five-token training backward also passes. Reports `runs/ling-completion-projection-check-20261002/{cpu,cuda}-report.json`. This does not certify full 8B memory/throughput. Future non-root containers must use a writable persistent `TRITON_CACHE_DIR`; the default inherited `/workspace/.cache/triton` caused a preserved permission failure before the successful retry.

Root generic trainer CUDA regression now **16/16 passes**, superseding the CPU-only 14-pass/2-skip check below. Sorted-dispatch reduced GPU/custom-LoRA prototype: upstream 4.316 ms, candidate 2.966 ms (1.46×), two-step optimizer state/gradients match; not a full Ling or real PEFT speed proof. Corrected right-padding comparison still has max logit difference 0.083984 above provisional 0.02; do not promote the padding patch. Detailed audit `runs/training-efficiency-20261002/report.md`.

v13-r2 preparation now passed: **111,301 rows =105,869 train +5,432 test (~4.88%)**, protected/reserved evaluation preserved. Final render/audit/mix stages continue. Local broader LFM350M experiment is being prepared from these reviewed model-neutral data, stratified modalities/families and source-group-separated heldout; GPU is released after the Ling tests. First-pilot turn diagnostic completed: adapter1/12 exact structural tool actions, base0/12,22/24 replies plain text without tools, no parse errors; `runtime-eval-turn-v1/report.md`. Luna1 has a reviewed64-case continuation running PID1921215 (40 source-backed non-TATQA,24 recovered, native64/64/309approveddecisions), replacing completed3-case recovery. Bunny275 completed269rawaccepted6reject; its next supply is being prepared. Luna2/DGX remain active, Bonsai remains disabled.

User specifically requested an efficiency audit, especially Ling's MoE implementation. Commit **db70db2** fixes `--expert-rank` matching individual routed experts such as `experts.12.gate_proj` while preserving stacked experts and the ordinary shared-expert rank. Actual PEFT regression verifies both rank and alpha. It also rejects partial activation checkpointing on custom models without per-layer support instead of silently retaining full checkpointing, and reuses the cached trainable parameter list for gradient clipping. Focused training-loss tests: **14 passed, 2 skipped because CUDA was unavailable in that test container**. Existing model/data/checkpoints preserved.

New Ling blocker: pinned custom forward projects every hidden-state position through `lm_head` and does not implement `logits_to_keep`. Our trainer supplies shortened completion labels, so previous reduced forward/backward with full labels did not prove trainer compatibility. The Ling agent is testing an isolated completion-projection patch against full loss and gradients; do not launch full training before that passes. Native short-training mode and padding-mask bugs are also under investigation; the current padding CUDA differential exceeds the provisional valid-logit tolerance and is **not** a clean equivalence pass.

Sorted MoE routing prototype now matches two CPU AdamW steps, gradients including absent-versus-zero patterns, and optimizer state with reduced frozen-base LoRA. Empty-expert zero dependencies avoid reading NaN weights. These are reduced CPU checks, not real PEFT GPU or full Ling throughput proof. GPU profiling and an exact-source-bound reusable patch installer are in progress; no sorted routing optimization has been promoted yet.

Authenticated pinned Ling BF16 download is running **without GPU allocation** on DGX user service `natlang-ling3-tiny-download-20261002.service`, external HF cache. Model revision `9a98e35fe1c9ee255f78dd64771c7ae15a799481`, 15,787,992,416 weight bytes. Status `/home/werg/natlang-remote/ling3-tiny-download-20261002/status.json`; last checked downloading, not verified ready. Qwen generation remains active. Do not load Ling on the DGX GPU while the teacher campaign owns it.

Full existing-data **v12 completed all 17 stages**, including token audit and reducer gate. A separate v13 improves the source-component holdout split; raw data and v12 are preserved. Exact corpus totals and final receipts are being collected. LFM350M whole-program v5 remains 0/4 before and after; the distinct 12-turn heldout diagnostic is still being executed and must not be called superseded by whole-program evaluation.

---

# Initial task evaluation and efficiency findings — 2026-10-02

LFM350M first100-step pilot is complete. Finalpairedwhole-programprobe runtime-eval-v5 uses4source-group-heldout originalIRs undersealed598c3 currentruntime, identicalbase/adapterseed and10turnbudget, explicitserveroutputcap512. Bothbaseandadapter0/4exactacceptance; two workflowcasesquiesced, arithmetic/CommitPackdidnotproducecorrectanswers/files. ExactreportreceiptSHA5d187cb6ae1bbe285b769d0bc3e5bf0d70f7b826ae2f5b4c4ac1bae153704ad1. Lowerheldoutlossisnotexecutiongain. Earlierlarger-budget/mainworkspace-dist probesarepreservedassuperseded, notpooledwithv5.12turn-leveldiagnostics/profilingstillrequested.

Efficiencybugconfirmed/fixedcanonicalb0e52ad: HTTPchat transport options.request.max_tokenswasoverwrittenbyruntimeperturnallowance. Nowtakesminconfiguredcap/remainingallowance,8transporttests+TypeScriptnoEmitpassed. Frozenactive598c3workersunchanged; student'sseparate--max-output-tokenscap wasneededforpairedfrozenruntimeevaluation. No language deadline introduced.

LingpinnedMoEtraining structurallyduplicateshiddenstates8x, does256maskscans/128expertPythoncalls, invokesgather/scatterexpertloop. IsolatedsorteddispatchcandidateCPUtestsexactoutputs/input+parametergradientvalues underfp32/bf16,notpromoted. BewareinactiveexpertgradNonevszero changesAdamWstate/decay;optimizerandLoRAtests required. SparkTorch2.13cu130F.grouped_mmavailableSM121, buttinycompatibilitysmokehitCUDAOOMwhilegenerationownsGPU; nofullspeedclaim/no liveinferenceconfigchanges. DeferSparkbackendcheckuntilgenerationreleasesGPU.

Continuity:Bunny275cleannewcasesrunningcurrent598c3,PIDs1886470/1886474,native275/275+550decisions. Luna1current598c3reviewed3case recoveryqueue launchedPID1893055;native3/3noholds. Historical50minutecheckconventionmustnotpauseunfinishedauthorizedsetup/queuehandoff. ExistingAPI/DGXgenerationcontinues;localBonsaidisabled.

Largecorpusv12split80heldout/111301rows duegenerateddefault explicittrainflags overridinghashpartition,notgiantfalsegroups. Newexplicit--repartition-unfrozen-train-groups recipeoptionwillretainprotected/test/frozenregistrysplitsandtransitivegroupclosure whilepartitioningdefaulttrainhints. Separatev13outputidentity; preservev12andrawdata. No fullproductionreadyclaimuntilfinalaudit/mix/splitqualitypasses.

---

# First real LFM350M training complete; task evaluation pending — 2026-10-02

Verified actual checkpoint/weights/adapter_config.json r=16, alpha32 and throughput args rank16. This corrects the earlier progress note's rank32 assumption (trainable count alone did not establish rank). Actual100steps/800examples completed with0skips, checkpointstateheldoutloss2.2986792588233946→1.3806172516942024. Approximately16minutes, peakallocated~6GiB withselectivecheckpointingabove4096tokens. Local modelhaslearned heldoutcompletionprediction inthisscopedpilot; whole-programtask accuracy isstillpending andnolargerstudentcomparisonestablished. Checkpoint+step100snapshot adapterSHAf1cd24b260a8e54dce491a839826c3b9b43fba5b81ac69d9020b319ce35bffa2. Six actualheldoutfullprogramcases and12turn-leveldiagnostics beingevaluatedbaselinevsadapterunderidenticalcontrols.

Lingtransportparser nowcommittedee84005,26focusedtests pass includingactualpinnedJinjatemplate roundtrip andexactargument preservation. Reducedkernelpadding/cacheequivalence stillpendinglocalGPUavailability. Bunny/Luna1queuescompleted; rejectionagentpreparingreviewedsuccessors. Full corpusv12streamingprepstillrunningboundedRAM, disk monitored (87GiBfreeatlastcheck).

---

# Real local student experiment running — 2026-10-02

LFM2.5-350M actual pretrained revision9e6c6ccf47cd318696e137d381a7ded8fe4df09f is training under runs/lfm25-350m-scoped-20261002/train-100steps. Current-policy re-admission48/48 teacher trajectories,451 decisions→413 approved/38 held; separate source-component split309train/104heldout (33/12 components). Trainreducers84/309=27.18%. LFM closed-template render/audit413/413, max14480tokens, no truncation. Real100optimizersteps×accum8 =800examples~2.59passes, BF16LoRA rank16 (5996544trainableparameters), selectiveactivationcheckpointing above4096tokens, recoverycheckpoint every20steps. Baselineheldoutloss2.2986793; step20checkpoint saved, finite losses and no overlength/skips, peakallocated5.9GiB. Posttrainingloss/task evaluation pending; no quality gain established yet.

Actuallength benchmarks: 2054tokens noactivationcheckpoint→two steps successful peak~3.4GiB;13293tokens noactivationcheckpoint OOM at configured6.86GiBcap; same longexample withcheckpointing→two steps successful~2.6GiBpeak. All413examples retained. ASR/voice check found noGPUvoiceprocess,18MiBidleVRAMonlyXorg; noCPUdesktop speechservices killed.

Ling pinned reduced KDA+MLA trueCUDA forward/backward withLoRA succeededseq65, notfull8Bspeed/memoryproof. Shorttrainingmode anddiscardedpaddingmask issues requireisolatedpatch/regressions. Transformers5.5 customcodecompatibility andLingXMLtoolcallparser/template audit ongoing. ActualBF16trainerdefault coversalllinear layers; restrictedq/k/v/o default is4bitbranchonly. Metadataonlymissingtokenizer was notupstreammissingartifact; officialpinnedtokenizer/template nowfetchedwithoutweights. DGXcontinuesQwenteacher generation.

Full existing-data DAG v11 admitted16524/16524resultrows, native materialized6GiB then concatenated124831rows8.4GiB; stale declaredinputpaths halteditbeforeprepare. Preservev11; correctedpipeline-v12.json/run-v12 is running under4GiBcap, preflightvalidatorregressionspreventthisearlier. Fulltests244pass2GPUskips beforeadditionalvalidatorchanges; focusedrunner20pass afterchanges. No fullcorpusreadinessclaim yet.

---

# Autonomous student experiments authorized — 2026-10-02

User approved the proposed local 350M-first / later DGX Ling approach and explicitly asked autonomous execution. Local experiment owner luna_quality_sweep: pinned real LFM2.5-350M BF16 LoRA, no gradient checkpointing initially, representative approved data, source-group-heldout baseline/after evaluation, memory/throughput and resumable reports. Existing-data full SFT DAG owner luna_source_oracle_audit: final strict positive-vs-provenance gates, full tests, bounded data-only build. Ling compatibility owner luna_oct2_rejection_review: pinned actual custom architecture reduced-config forward/backward, adapter coverage and formatting; preserve Qwen DGX generation until complete.

Root pinned Ling metadata/code at revision9a98e35fe1c9ee255f78dd64771c7ae15a799481 under runs/ling-student-compatibility-20261002/source-manifest.json. No full Ling weights or production training launched. Short-sequence training mode assertion in upstream custom code is a candidate bug under investigation, not a proven installed-runtime failure yet. Do not conflate prior tiny random diagnostic with the authorized real pretrained-model experiment.

---

# Student experiment direction — 2026-10-02

User proposes a local LiquidAI/LFM2.5-350M experiment and likely targets inclusionAI/Ling-3.0-tiny for DGX student training after generation finishes. Ling is a preference, not a finalized training configuration; no production training has started and DGX remains on Qwen teacher generation.

Recommended local experiment: bounded, source-group-separated approved-data SFT with before/after runtime task evaluation, representative primitive/file/directory/delegation cases and the 25% reducer target. Start BF16 LoRA with gradient checkpointing disabled, then measure real memory/throughput across sequence lengths. Do not truncate or discard long trajectories silently to achieve a fit. Periodic disk recovery checkpoints remain distinct from activation/gradient checkpointing. A 350M quality ceiling must not be treated as evidence that Ling cannot learn the tasks.

Ling-3.0-tiny official model card reports 7.9B total/1.3B active parameters, hybrid KDA/MLA and routed MoE. Active parameter count does not determine full training weight/optimizer memory. Verify differentiable kernels, adapter coverage, template/tool/reasoning formatting and heldout runtime evaluation before a DGX training commitment. Keep model-neutral approved data and render separately for each student tokenizer/template. User discussion does not authorize stopping current DGX generation early.

---

# Storage block removed — 2026-10-02 18:53 UTC

User authorized freeing/reorganizing storage. Reclaimed aged unused Docker build cache and untagged images unused by containers (older than168hours). Actual filesystem available space rose from10.63GiB to153.48GiB, increase142.85GiB. Receipt runs/storage-reorganization-20261002/receipt.json; raw Docker accounting totals include shared layers and are NOT the actual free-space increase. Tagged natlang-train image remains exact eabc88d83cba79860c368bd2c7758e0e6914f602e7de6b9e20446fb624de1cdc. No training data/model files, containers or volumes were removed/moved. Remote DGX external has544GiB free; no storage transfer was necessary for this block.

Bunny current256 completed256 exact exports,252 raw accepted4 rejected. Its follow-on is now running PID1796554/supervisor1796566, last38 exact accepted results. A gated launcher first failed BEFORE model calls because canonical runner lacked --provider-request-config; preserved setup failure and retry uses pinned compatible v35 helper. Canonical runner support is now ported and tested16/16, commite728e95; future launch plans should use its updated SHA rather than rely on hidden run-local feature patches. Luna1 fixedx64 active PID1767281 (firstcase70turns, correctfalse andcurrentadmitted), Luna2 remains active1383567; DGX generation continues. Old Bunny211 stays unresolved/raw and covered by existing recovery assignments.

Standalone and workspace dependency lock fixes are committed46fb511 and2702b7e. Workspace adds only25 exact optional esbuild0.28.2 platform records, keeping existing locked entries; clean offline installs validate x64 and LinuxARM64 dependency resolution. Main node_modules/live worker runtimes were not installed into.

Full-data SFT v33 DAG now exists under runs/data-lineage-20261002/recipe-v33-current-policy-existing-data-only/pipeline.json. Historical resource guard recorded55,014,686,843bytes required versus11,410,812,928available and stopped beforematerialization; this storage block is now resolved. Final inventory/role classification and CLI graph validation remain in progress: source/hold metadata hashes must be carried visibly without being marked positive training inputs. Do NOT bypass required-default/quality gates or claim a full production run complete yet. DPO61 unique causal pairs are explicitly tracked in a separate pending-render/split/audit lane acrossfourphysical files, not SFT inputs or automatic negatives.

---

# Active generation and infrastructure repairs — 2026-10-02 18:32 UTC

Controlling instruction is unchanged: only Bonsai is disabled; keep Luna and Space Bunny generating and use this machine for training-pipeline testing. DGX Qwen generation remains active with the951 fixed-v38 successor gate waiting for predecessor completion.

Actual local workers: Space Bunny PID1754718 on the256 current quality-v4 workflow queue; Luna1 PID1767281 on20 reviewed current workflow recovery cases using fixed x64v38; Luna2 PID1383567 continues306 prior reviewed cases. Luna1 first launch1765401 failed before model calls because collector log parents were absent; creating parents allowed restart of the same uncompleted key. Canonical runner now creates jobs/output/log parents before recording/starting the attempt, commit2343bb6;14 queue tests pass. No setup failure is a model negative. The old Luna2 runner has no graceful drain marker: don't terminate an active healthy case to change runtime or fabricate completion journal records; fixed runtime can be used for its reviewed successor.

Space Bunny's fixed v38 runtime was blocked by a hidden npm lock mutation caused by a private build hardlink clone. Preserve corrupt bytes/receipt at runs/runtime-integrity-investigation-v38/. Exact DGX copy restored LOCAL v38 only, with full20,179-file/22-link verification. v37-r3 remains integrity-invalid and quarantined. Safe physical clone helper and five regression tests committedcca6938; incident audit runs/runtime-hardlink-isolation-audit-20261002/audit-v2.json. Never run package managers in a hardlinked sealed runtime derivative. New fixed x64 v38 closure independently root-verified20,180 files,22 internal symlinks,all hashes/modes,zero hardlinks; manifest e16ded63d78e023ff2e49b0ffcc92fab7baeed2eb90349fc1e02911f4c1afc86.

Full Python suite229 tests passed in63.28s (eight existing warnings), receipt runs/training-pipeline-preflight-20261002/python-suite-229-review.json; subsequent helper/queue focused tests also passed. The standalone ts-host package lock now matches unchanged package.json and preserves all original locked versions plus25 cross-platform esbuild optional entries. Clean offline npm ci succeeded in a private clone; root promotion46fb511, lock SHA85380ae3ef7e5cfb0b3c450e00b5e6991959ef8063eb38348ebfd1005ab8c6b6. Main dependencies/live frozen runtime trees were not installed into or rebuilt.

The first new Bunny rejection is a semantic invoice-rule error, not demonstrated infrastructure or formatting failure. It ignored an explicit single-unit lump-sum exclusion; raw attempt remains rejected and no oracle weakening/gold injection/DPO assumption is made. Receipt runs/space-bunny-qualityv4-next-v38-20261002/result-review/first-rejection-review.json.

Current full-data recipe is moving to recipe-v33-current-policy-existing-data-only with a physically independent current-source runtime. Initial v32 runtime clone also had source/lineage hardlinks and must not be treated as isolated. New seal verifier rejects aliases and hashes nested lineage manifests. Executable DAG/resource guard evidence still pending at this timestamp; local disk about11GiB means full materialization must wait for storage. Do not claim production corpus/student training is complete.

---

# Current controlling scope and verified progress — 2026-10-02 18:15 UTC

Stop only local Bonsai generation; continue Luna and Space Bunny locally or on DGX. Test and fix the training pipeline locally. Do not restart Bonsai without an explicit user instruction. Its durable disable marker remains active.

Luna2 PID1383567 continues its306-case queue (case135 active at this check). Luna1 short32 finished32 exact exports,23 admitted9 rejected. Space Bunny short33 finished33 exact exports,28 admitted5 rejected. The old300 Bunny queue has only299 exact exports: index211 is an unresolved incomplete attempt, not a completed training positive or DPO negative. Its fresh recovery is already covered by DGX/Luna2 assignments.

Prepared successor packets: Bunny256 and Luna1 20 current workflow quality-v4 cases, with exact gold/admission/conversion/native proofs and zero source holds/unlinked decisions. They have NOT launched: launcher correctly detected a changed hidden node_modules lockfile in the local sealed v38 candidate. Preserve changed bytes and trace provenance, restore only an independently verified matching file, then verify the entire closure before launching. Luna also needs a sealed fixed x64 derivative. Do not bypass integrity checks or claim idle slots are running. Keep Luna2 active until a safe boundary transition can use the fixed runtime.

DGX current1024 queue remains active; latest verified525 exact results,307 partials,0 errors. The fixed successor951 gate IS armed, enabled and waiting: natlang-bonsai-qwen-successor-gate-v38.service, PID969013, receipt runs/dgx-bonsai-qwen-handoff-20261002/gate-v38/gate-arm-receipt-v38.json. Old v37 gate disabled. Fixed ARM v38-r2 was fully verified on DGX;951 exact native/current-policy compatibility proofs pass. Gate must wait for all predecessor exports and process/lock drainage before launch; no951 model calls yet.

Published current static directory v26:1212 raw cases,1199 eligible13 explicit holds;4556 decisions3596 approved960 held. Qasper reference evals now visibly expose UTF-16-correct original source spans; annotation-directed scripted construction is explicitly recorded,300 unsupported direct returns remain held. Seven persistent tests pass. Publication/source fix commit9820f58. Prior raw data and superseded manifests remain preserved.

Historical repair selection was repeating work:30 of32 Luna repair cases already have current eligible-positive results. Current-version guard and four tests committedecedf40; it reproduces registered transformations and preserves source/files/golds before history checks. Never relabel old trajectories as new versions or suppress from unvalidated raw acceptance alone.

Training checks: full Python222 tests passed;413 actual approved examples rendered/audited with2,376,629 total and64,855 supervised tokens; tiny random diagnostic CUDA training/checkpoint smoke passed. This is not production student training. Serving had an additional stop-token mismatch: generic MiniCPM EOS differs from closed-template im_end130073. Commit35cbf4f binds generation to the validated template token, strips only its final occurrence, preserves tool syntax/token usage and tests finish reasons;10 focused tests pass.

Full existing-data executable DAG and isolated current runtime/dependency closure are being completed. The initial v32 JSON spec was declarative, not executable. Campaign v38 differs from current repository policy in5/7 modules and is NOT a substitute for an all-corpus current runtime. Current package.json/package-lock mismatch breaks clean npm ci and must be repaired in an isolated coherent clone before canonical lock promotion. Local disk about11GiB free; full-data preflight must run first and fail with a durable resource-wait receipt before large materialization. Do not call the full training corpus ready or delete bound raw snapshots to free space.

---

# Confirmed runtime cleanup fix and continuation — 2026-10-02

Controlling user correction: stop only local Bonsai; continue Luna and Space Bunny, and test/fix training locally. Bonsai disable marker remains active. Luna1 repair32 exported32/32 exact, Luna2 exported95/306 at latest root check and continues. Prepare reviewed next Luna1 supply; do not describe an exhausted queue as running.

Confirmed delegate lease leak is fixed in canonical source, commit80c78f5. Kernel now aborts every supplied/acquired folder transaction on any invocation setup/preflight failure; delegate schema explains valid Natlang result types. Isolated typecheck and90 runtime/interpreter/tool-surface tests pass; earlier full Python222 tests and real413-row renderer/audit plus tiny GPU gradient/checkpoint smoke passed. Runtime deployment must use a newly hash-reviewed sealed snapshot; no old frozen tree is changed.

First newv38 candidate was accidentally based on unsealed630-file scratch, not approved ARM37r3. Rejected BEFORE deployment/model calls; receipt runs/space-bunny-directory-v7-20261002/runtime-v38-candidate-rejection.json. Correct replacement must derive exact ARM37r3 manifest0778802aa821e018c8d4a3c1b5e81e84ba45b9ab32d551d53d8096481398894c, retaining20,179-file dependency closure and ARMNode/esbuild. Bunny33 credential-preflight failures are retained as zero-model setup failures; actual authenticated launch waits corrected snapshot. Existing owner-only OpenRouter env file is used, never print key.

DGX current1024 inference remains active; latest463 result files,303 partials,0 errors. Its951 waiting successor gate is deliberately inactive pending new reviewed runtime; pause receipt runs/dgx-bonsai-qwen-handoff-20261002/gate/gate-pause-receipt-runtime-fix.json. Preserve old approvals and trees, reprove951 on actual ARM then rearm a versioned gate after review. Old Bunny211 partial stays unresolved and is already assigned to active DGX1024 and Luna2306, so no duplicate new attempt now; no SFT positive/DPO negative for incomplete infrastructure output.

Static directory publication v26 is installed with canonical pointer SHA d51ad1c64f26f093c55c6f7ef45f689cf08f3ddaef23dd63001dcaa83bb22367:1,199 eligible cases +13 explicit source holds,4,556 turns/3,596 approved/960 held. All300 Qasper references show original cited text,300 direct returns remain held. Previousv25 and all source records/golds retained; catalog merge/receipt finalization ongoing. Full final student corpus is NOT training-ready; full data-only recipe refresh and disk capacity remain required (local~14GiBfree). No production student training launched.

---

# Space Bunny completion correction — 2026-10-02 16:42 UTC

Root independently inspected all300 aggregate outputs and found **299 results (292 raw accepted, 7 raw rejected), plus index211 incomplete_export**, not300 exact results. Earlier last-job-based completion statements below are superseded. Supervisor correctly recorded index211 partial_without_terminal_result (exit0, no final manifest, two saved model responses); process ending does not prove campaign completion. Authority now records stopped_incomplete_export with retained211partial, no positive/DPOnegative for it.

Index211 second reply invokes two `delegate` calls on the same `jobs` folder; there are no child provider request observations before Node exits0. Confirmed folder-lease leak: delegate acquires its transaction, then malformed prose returns fails type parsing before kernel cleanup; the next same-path delegate blocks. Deterministic no-provider reproduction: runs/delegate-deadlock-audit-20261002/audit.md. Source cleanup/schema regression fix is in progress; frozen versions must stay immutable. Do not blindly resume/mutate frozen36 or claim a successful handoff. New source-valid disjoint failed-only cohorts are prepared: Bunny33, Luna1 32; current Luna2 and DGX generation remain live. Explicit transfer/recovery accounting is required before substituting a completed predecessor claim.

---

# Final streaming regression pass — 2026-10-02 16:37 UTC

Recipe now defaults to memory-bounded data stages; `--in-memory-data` is a diagnostic opt-out. Stable rehearsal tie ordering matches the old builder even for repeated IDs with different content/difficulty. Full Python suite **222 passed** in57.38s (8 existing deprecation warnings). Streaming token audit's storage guard passed too. Canonical production files are committed; static catalog refresh and local generation successor preparation remain active.

Root independently checked all300 Qasper reference candidates: original source paper files/golds unchanged, each printed cited block exactly matches the source range, and all answer spans are visible after reassembling the cited chunks. Receipt `runs/generation-check-20261002/manual-0734/qasper-visibility-review-v2/root-visible-evidence-review-v6.json`; this receipt does not itself publish the static bundle. The canonical builder must use UTF-16 offsets consistently (string.slice, not Array.from indices), label original-human-annotation-selected reads as scripted reference construction, and retain prior raw trajectories.

---

# Memory-bounded training pipeline — 2026-10-02 16:30 UTC

Implemented opt-in `--streaming-data` in the training recipe: SQLite-backed preparation and joint assembly, chunked rendering, and shard-backed token audit with lightweight dedup/split metadata. Default semantics are preserved; full bodies remain on disk. Focused parity/resume tests pass, actual 413-record MiniCPM rendering and ready corpus are byte-identical to ordinary paths. Full Python suite **220 passed** in58.51s; a subsequent audit disk-guard regression also passed (24 audit tests). Existing8 Spark/Transformers deprecation warnings remain.

Storage guards fail before large scratch work: preparation5x input+2GiB, assembly2x+1GiB, rendering3x+1GiB, audit4x+1GiB. These are conservative planning bounds, not guaranteed upper bounds under arbitrary expansion or concurrent disk consumers. About12GiB local free remains; full existing-data production build not launched yet. Smaller real-data renderer/audit/gradient/checkpoint paths have completed. Full data-only DAG is being pinned; static publication also waits the Qasper visible-evidence repair/replay. Preserve raw input corpora and prior pinned proofs.

DGX current1024 pool continues; 951-case successor gate armed/waiting as recorded below. Space Bunny finished all300 exact queue entries around16:23:44UTC, Luna1 finished256; successor preparation is active, Luna2 remains running. Do not claim both local slots/Bunny running during normal queue completion gaps. User requested these continue; feed reviewed successors without restarting Bonsai.

---

# Local training pipeline fixes and generation continuity — 2026-10-02 16:11 UTC

The user's correction remains controlling: only local Bonsai is disabled. Luna2 and Space Bunny remain live locally, DGX Qwen remains live. Luna1 completed all 256 exact exports normally; authority now records completed rather than stale running. Bonsai disable marker/watcher/server guards remain in force; local GPU is reserved for training work.

Actual-data training smoke used 48 accepted teacher trajectories across 29 family/modality groups. Current materialization produced 413 approved decisions and 38 held. MiniCPM tokenizer's generic EOS `</s>` differs from its chat template assistant terminator `<|im_end|>`. The old renderer discarded all 413 valid examples and returned success. Renderer now derives/validates the registered closed-template terminator; readiness/audit bind it, and all-invalid renders preserve rejection evidence but fail. Pipeline render stages explicitly require nonempty JSONL. Actual fixed render/audit has 413 ready rows, 2,376,629 total tokens, 64,855 supervised tokens, max length 13,828 under 16,384. Evidence: `runs/training-pipeline-gpu-smoke-20261002/minicpm-fixed-v4-r2.ready.jsonl` and manifests; runner smoke `runs/training-pipeline-smoke-20261002-v2/` completes render/audit.

A tiny random diagnostic Llama trained on four actual approved records for two CUDA steps, finite gradients/loss and saved/reloaded checkpoint; peak GPU memory 307 MB. This verifies the data/loss/checkpoint path, not a production student evaluation or student selection. Report `runs/training-pipeline-gpu-smoke-20261002/actual-data-gpu-step-report.json`.

Focused Python tests passed (73 renderer/readiness/audit/mix/checkpoint/loss/DPO, 33 pipeline/recipe, 13 queue, 5 freezer). Full Python suite now passes: **213 passed** in66.63s (8 existingSparktransformersdeprecationwarnings). Invocation usesOMP/OpenBLAS/MKL1 andpidslimit1024; fakeGPUprobe test usesPython shebang becauseimageBashstartuphook recursivelycallednvidia-smi whenmockusedshell. Suite ran in existing natlang-train Docker with host Node24 bind, user1000. Some recipe unit fixtures accidentally scanned/wrote the real corpus; isolated fixtures now prevent this. Memory-bounded SQLite preparation andchunked rendering implemented, parity/resume tests passed; jointassembly still being checked beforefull-data run. Do not blindly execute the old all-lane recipe here.

DGX successor selection/native proof: 951 cases = retained 907 backlog +44 unfinished Bonsai cases uncovered by eligible/current/pending Qwen. All original IR/golds preserved, new Qwen attempts, no cross-model partial continuation. Full ARM37 exact replay/admission/materialization proof passed 951/951. Root found and corrected draft runner/Node pins and final-status gate schema/race issues; remote waiting gate is armed at16:14:40UTC (unit natlang-bonsai-qwen-successor-gate.service, PID890606); root independently verifiedwaitingstatus/activeunit. Receipt `runs/dgx-bonsai-qwen-handoff-20261002/gate/gate-arm-receipt.json`. Do not claim successor launched until remote receipt proves it; current1024 pool must finish exact exports and release its process/lock first.

Correction-round builder nowusesbase repositoryandapproval-bound legacyidentitystages; previouslyhardcodedrealrepodiscoveryandduplicatelegacyIDs crashedcombine.

Static refresh: 1212 modern directory cases retain 1199 current-policy eligible and 13 explicit source holds, including 17 MuSiQue +3 TATQA reviewed answer variants. Currentstatic-bundleprovenancevalidationpassed1199, butrootreviewcaughtQasperapprovedexplorationreading/discardingpaperbodies. Sourceagentrepairingreferenceevaltoprintactualtextandreplayall300; answerreturnsremainheldwhenunreasoned. Publicationwaitsthisqualityfix; old candidate receipts retained. No source hold is a model negative. Source agent also tracks 9 selected teacher rows not materialized: 7 retired checkpoint-phase records, 2 incomplete-ledger-linkage records. Current-policy eligible versus raw labels must remain explicit.

---

# Bonsai stopped; local training pipeline testing begins — 2026-10-02 15:27 UTC

User initially requested moving all local generation, then explicitly corrected: **stop only Bonsai**. Keep two Luna workers and Space Bunny running here, DGX Qwen continues. Local GPU reserved for testing/fixingtrainingpipeline onexistingdata. Do not stop/move Luna/Bunny merely because earlier instruction wasbroader.

Bonsai supervisor778887/collector1456382/armedcontroller1324745 stopped; first Dockerstop triggeredwatcher368408 restore race, caught andwatcherstopped/restoredDockerstopped. GPU18MiB/0% afterfinalstop. Dockerrestart=no. Durablemarker `runs/bonsai-generation-disabled.json`; watch_bonsai.sh exitswhilemarkerexists andserve_bonsai.sh refusesstartup. Remove markerONLYwhenuserexplicitlyresumesBonsai. DisabledtwoBonsai-onlystorage-recoveryapprovals underauthoritylock, retainedothers. Finalstopreceipt `runs/bonsai-stop-training-pipeline-20261002/final-stop-receipt.json`. Preserveallraw/completed/partialoutputs; unfinishedsourcecases+approvedBonsai256backlog transfer toDGXQwen withnewteacherattempts, no crossmodelpartialresume. Luna rejectionagent ownsremotequeue/nativeARM37proof/handoff; currentDGX1024notinterrupted, reviewed907backlogretained.

Trainingwork: rootownsGPU/environment/readiness/checkpoint/loss tests; Lunaqualityagentowns trainingpipeline/config/tests andconcretefull-current-inputrecipe; sourceagentowns nine materializerrejections andstaticcatalogalignment. Tests nowexplicitlyauthorizedbyuser. .venv lacks torch; useexisting `natlang-train` Dockerimage, no unnecessaryhosttorchinstall. Roottorch-independent render/auditcorpus/mix tests33passed1skipped; readiness/checkpoint/loss/DPOdocker testsstarted. Existingsharedtraining/data_sources editsandMuSiQueprototypefilespreserved; nofullcanonicalTypeScriptemit whilelocalteachersrun.

Inventorycorrection fromcompletedagentreport: CURRENTpolicy-approvedstaticdecisions32,615, notrawlabels32,952;105directoryrowsheldsource_review_pending. Currentgenerated74,666+static32,615+reference5,174=112,455approveddecisioninstancesbeforecross-lanededup (+125reviewedlegacyrecords). Roughpre-renderestimate≈726Mchars/4tokens, notexactstudenttokenizer/finaldataset. Report `runs/training-inventory-20261002-1508/inventory-audit/current-inventory.json/.md` supersedesinterimstaticcounts; nine selectedbutunmaterializableteacherrowsstillbeingtraced.

---

# Full training inventory — 2026-10-02 15:18 UTC

Independent current admission snapshot `runs/training-inventory-20261002-1508/generated-snapshot/b8deb585-b207-48dd-9a59-eea8793f5b4f.manifest.json` (cutoff15:09:13) selects14,332teachertrajectories/6,428exactprogramIDs: Luna5,716/Qwen3,922/Bunny2,914/Bonsai1,499/Horizon147/older134. Native materialization measured14,323accepted,9rejected (2unlinked),74,666approveddecisions/5,280held; sourceagent auditing9, preserve selection/materialization distinction. Teacherfamilycounts9,951workflow/3,616source-backed/765other.

Generation-target coverage uses6,138canonicalsource/workflowcases:5,069everattempted,4,833witheligibleteachertrajectory,236onlyrejected,873unattemptedunheld,196unattemptedheld. Additional2,192inline reference-onlycases already have deterministictrajectories and5,174approveddecisions; includingtheminnever-teacherattempt count gives3,065unheld, but873is the actualsource-generation backlog. Variants/pairedreducers outsidecanonicalexactIDs and cross-teacherretries are separate, notfreshsourceclaims.

Provisionalcurrentstream inventory112,792approveddecisioninstances +125reviewedlegacyprogram-editingrecords, beforecross-lanededup. Roughchars/4 tokens: generated507.8M/static204.3M/reference15.9M≈728M prompt-plus-target. Teacheranswers+reasoning≈12.5M targettokens. NOTtokenizer-exact/finaltrainingready. Staticpublishedrawfiles6228cases/32952approveddecisionlabels differcanonical6138 (directory1302vs1212);90extras needexplicitcatalogalignment, agentreviewpending. Rootinterim receipt `root-interim-inventory.json`; detailedagentreportpending. Recipe-v31 old4037-teacher snapshot is stale; no newdefaultrecipe/finaltrainingpublication made bythisinventory.

At15:18DGXimportledger309=305admitted4rejected;GPUpreviouslive96%;Bonsai/Luna1recovered/Luna2/Bunnyactive. Fledgeblockedproviderregion (permanent403), no usableoutput. Next907DGXreviewedqueuepreparednotlaunched. Prior50minsleep interruptedbyuserwork; resumeactualsleepafterinventoryquestionandurgentchecks, neverclaimdurablechatwakescheduler.

---

# Live inventory sweep and worker corrections — 2026-10-02 15:07 UTC

- Full inventory/unique coverage/token estimate is being reconciled against current results; recipe-v31 generated snapshot is Oct1 15:58 and must not represent all current generation. Its audited pre-render inputs total64,384 approved decisions before final dedup, including6,138staticcases/32,652staticdecisions. New DGX/local outputs require current admission and a new snapshot; native reference proofs are not additional model generations. Inventory reports pending from Luna auditors.
- Actual Luna1 successor1376317 died before any model request because its collector log parent directory did not exist. The earlier authority said running; this was incorrect. Root reverified all659pinned artifacts, created all queue log parents, restarted SAME reviewed queue/runtime/journal with no completed cases lost: actual1435328 at15:04:49; first exact trajectory completed15:05:00 with one model reply. Receipt `runs/teacher-successors-v33-20261002/luna1-successor-v33-20261002/root-log-directory-recovery.json`. Canonical supervisor is still pinned by armed controllers; don't edit it in place. Future launch preflight must create/check all collector log parents and verify actual worker progress rather than trust start receipts.
- Fledge Alpha Free on OpenCode Zen was attempted with user-saved owner-only key; verified free catalog and structural model profile. Both workers failed before any model reply with HTTP403 `FreeTierError: This model is not available in your country`. Five persisted zero-reply infrastructure failures, no training positives or DPO negatives. Parent1422407 and both supervisors1422422/23 have stopped; NOT generating. Preserve original reviewed launch plan/pins/journals. Provider regional eligibility is required before a new versioned attempt; no paid fallback. Source agent preparing durable blocker evidence and a permanent403 first-error circuit breaker.
- DGX current1024pool remained active at96%GPU;15:05 local imports290=286admitted+4rejected. SpaceBunny current300queue147complete; Bonsai513-case queue73completed BATCHES of129, don't subtract batch counts fromcasecounts. Luna2actual1383567 continues306queue.
- Future DGX reviewed backlog is907cases (205reducers702primitives), not earlier723 estimate; exactdisjoint from current local queue scope, historical source reuse disclosed. Current1024+future907 combined reducer share26.15%. Runtime37ARMcandidate r3 complete closure root-hashchecked, stillnotdeployed/remote-executed.
- User's new Fledge work/status question interrupted prior sleep; previous14:50deadline has elapsed. No claim of current sleep or independently scheduled chat wake. Resume actual50-minute sleep after current urgent corrections and inventory answer.

---

# Metadata proof corrected; actual sleep starts — 2026-10-02 14:00 UTC

Root independently checked all306correctedselectionrows againstoriginals: onlyhistoryflags/membershipmetadata changed, all3,584referencedmembershipfiles agree withactualsavedIR. Exactlyonepriorprogramoverlap (authorizedfailedretry); all306havehistoricalsource/groupoverlap. Correctedv2proofSHA3bcb8f8482c7b00e4a2e803b2bb17ea887e8b55bfb95c2202b8a524ff37933e0; rootaddendumSHAdb232c760e3e3acdcdf4892dfea26fbf87e99a14bf18726cf2debc14c79717c2 nowboundactualLuna2authorityworker/launchrecord. Originalpinnedproof/IR/runtime/queue preserved. No publicationcausalDPOclaim. Prior13:58sleepentrywaspreparedbeforeurgentcorrectionarrived; ACTUAL50minuteclock.sleepstartsnowdeadline **2026-10-02 14:50:37 UTC**, active-sleep.jsonupdated. Resumeabsolutedeadlineafterurgentinterruptions.

---

# Worker supply restored; next active50-minute check — 2026-10-02 13:58 UTC

- Actual Luna2worker1383567 nowrunning306 reviewed36cases; controller1383562/rootplanSHA77de00963d722fdfd265bf18703304f345f77a4eaefb56e86a735c4c3b5d35c3; all659pins/native306IRexact/allaccepted/6047decisions verified. Runtime/program/goldsunchanged. Agent's per-rowselectionproof defaultedbothpriorLunaID/sourceoverlaptofalse; aggregatereportcorrectlyrecords1explicitretry/596sharedpairsourceIDs/5sharedTATQAsources. Rootlaunchedbeforeagent'slatewarningarrived. Metadata correction requiredbeforepublication/inventoryuse: agentpreparingseparatev2proof, originalapprovedpins/rawworkerpreserved. Do notinterpretoldproofasfreshsourceclaim orautomaticcausalDPOpair.
- Allteacherscurrentlysupplied: DGX256requestgate/320activecases; Bonsai778887 long513queue+armedreviewed256successor; Luna1actual1376317/new256queue; Luna2actual1383567/new306queue; SpaceBunnyactual1379568/new300queue. Verifiedlatest-worker-supply.jsonreceipt.
- Production70.59secondssample333.46outputtokens/s,6736prompttokens/s countedbyvLLM (promptcounterincludescacheuse; notphysicalprefillthroughput). Earlier110seconds475.23outputtokens/s. Workloadvarying; do notclaim475sustained/fullcampaign. KV57.2%,251active atnewsample, no reportedresourcefailure. Completedartifactmixstillmostlyprimitives, reducerqualityconclusionspending.
- DGXnextbacklogconservative723candidate=687freshunheld+36explicitreviewedrepairretries,376reducers52%; countssupersedeearlier853afteradditionalactual/plannedLunacohorthistoryfilters. Stillpreparingnativeproof, notqueued/deployed. Future37completeclosure+supportedARMNode preparationonly, live34unchanged.
- Startingactual50minuteclock.sleep now; absoluteUTCdeadline **2026-10-02 14:48:42 UTC** persistedruns/generation-check-20261002/active-sleep.json. Handleurgentnotificationsthenresume SAMEdeadline. No independentagentwakeschedulerclaim. Onwake checkservices/imports/resourcepressure/newrejections/workerqueues/correctionproof/nextqueue.

---

# Space Bunny300 launched and Luna history correction — 2026-10-02 13:50 UTC

- Actual newSpaceBunny300directory-v7 worker isrunning: launcher1379473/supervisor1379568, immutable36. Rootreviewall630runtimefilehashes/659preparedpins/3284providerdependencyfiles and exact256actualpredecessor result/output/IR/trace comparisons. ActualrootrolloverSHAfa406adb4c83f39f08916a2bdb292bd58ff15509e0dfc685f6f0527034a1a1b3; workerSHAe8c7420a452f4f76bff20ca579c89ab56bf11d282688dc96ccee852d29c7a260;665rolloverpins. Firstnewcaseaccepted,8turns; capturedcontextcontainsnewclaim-to-evaluateprompt, provenancefreeStealthonly/no distillationflag. Currentprior256was255accepted/1reject, notolder188/68. Actualpredecessorplanv35-v2 used inrootrollover (agentdraftnamedv1). SourceID/programoverlap0;9sourcegrouplabelsoverlapexplicit. Preservedwrongdrafts review-history. Local36doesnotsealallNodepackages; knownparent-dependencylimitation recorded, verifiedprovider3284pins only.
- BroaderactualsavedLunahistory supersedesearlier14/17nonrecentclaims: all12workflowrepaircandidates hadpriorLunaresults (11accepted,1rejected); fiveTATQAonlyappearinretired-before-launchqueue/noresult. NextLuna2candidate now306=5unattemptedTATQA+1explicitpriorfailedworkflowretry+300cross-teacherpairreducers. Rootauthorizedthisshape; native/rootapproval/launch stillpending. No falsefreshnessclaimforretry/sourcequestionreuse.
- NextDGXselection conservativecurrenthistory811fresh/unheld (356reducers43.9%), plus42nonoverlapreviewedrepairretriescandidate=853. Other25repairIDs overlapcurrentlocalqueuesandexcluded. No2048quantityquota; quality/sourcecoverage wins, continuouslysealedbacklog. Nativeproofnext/policies pending, selectioncounts mayshrinkonproof. Sourceagentpreparingcompleteclosure37fromexact36andisolatedsupportedARMNode, no live34/serverchanges.

---

# High-concurrency production check and Luna handoff recovery — 2026-10-02 13:44 UTC

- DGX central pool still healthy: 253 active requests/1 waiting, GPU96%, KV56.8%, zero preemptions, 26,720MiB available RAM. Latest128 imported artifacts126admitted/2wrong_return rejects. New invoice rejection asks whether document is NOT an invoice; observed document explicitly INVOICE, model returnedtrue ratherthanfalse. Supported gold, no format/parser defect found. Prior customer pair also reversed boolean criteria. Raw failures preserved. Production sample475outputtokens/s remains short-interval evidence, not full-campaign benchmark.
- Luna1 predecessor256finished. Its automaticcontroller1308753 had exited during transient full-TypeScript-emit source-review.js byte mismatch; restoring reviewedbytes did not resurrect it. Root reverified all659pins/exactpredecessoraccounting and created new recoveryplan/record, preserving old crash. Newcontroller1376313 successfully launched actual Luna1worker1376317 with256reviewed33cases. RecoveryplanSHA1ad6f74202e806959f93d1d77583c4fff30f6b94e3c0f131dc5c4960a0834856. Bonsai controller1324745 still alive; no duplicate workers.
- Luna2 current27finishedexact27/27; next314beingprepared (14unattemptedsource-reviewedrepairs +300directory-v7cross-teacher cases). Three previously successful exactLunacasesremoved; fiveTATQAfoundonlyinretired-before-launchqueues mayproceed. No successorlaunchclaimyet.
- Root caught wrong predecessor in SpaceBunny300draft: it audited oldspace-bunny-reducer-successor, not actual finishedspace-bunny-workflow-next5-v35. Old188/68stats MUST NOT be attributed to current256. Correctedpacket/predecessorreceipt pending. Root independently checked all630runtime36files, exact35base outside5reviewedprompt/runtime paths, and actual300native rows matchingselectedIR; allaccepted,6311turns/6011approveddecisions. Root-native-check underbundle. Existing drafts preserved, no launchuntilcorrectactualbinding/allruntimepins.
- Source agent preparingcorrectedBunnypacket; qualityagentpreparingLuna314; rejectionagentpreparingnextDGXbacklog and reproduciblev2→v3annotationtransform. No live34mutation, no unreviewedtrainingpublication. Nextactual50minutesleep follows urgentworkerreplenishment.

---

# Pipeline review and disk moves complete — 2026-10-02 13:24 UTC

- DGX production pool continueswith256requestgate/320cases; latest local65imports alladmitted, noinfraerrors atlatestcheck. Production110s sample475.23outputtokens/s while254requests active; shortinterval only, notwholecampaignbenchmark. Signalwarningaudit saw913warnings/max326listeners,343starts23finishes=>320active,95%GPU,0errors/retries; boundedfan-out plausible, no demonstratedrunawayleak. Canonicalfuturetask.close nowabortsprivatecontroller AFTERdrain; completed-task signal becomesaborted (lifecyclechange). Canonicalfuturepoolerrorcleanup retainsPopenhandle,TERM+wait10seconds, KILLonlyifexceptioncleanuptimeoutexpired,wait/reapbeforelockrelease. No live34/supervisorremote mutation. Proof `task-close-fix-review/report.md`;8nativecases underlocalNode24passed, not320requestdeploymentvalidation. Frozenfuturedeploymentstillpending.
- Genericcanonicalprompt nowtreatsdeclarativebooleanpropositionsasclaimstoevaluate, notfactsassumed. It follows explicitlyspecified missing/no-match taskresults (successfulnegativepossible), blockswhennecessaryinputmissingANDnotaskfallbackdefined. Thisfixes conflictbetweenearlierblanketblockguidanceandinvoiceap00106explicitno-pricefalsecriterion. Canonicalcompiledsource/JS updated/typechecked; notyetinlive34/33/35. Source/gold unchanged; no broadbooleanoraclechanges. Agenttracefield-locationguidancecandidatepending.
- Final2048delta13review v3 supersedesinitialoverholdrecommendations:ap00292mismatchedstickerquantityvsdifferentlineisintentionalfalseproposition, no sourcehold;ap00106definedfalsefallbackbutpotentialsystem/taskpromptconflict, notsourcehold. No new sourceholds. Keep originalreview v1/v2; noautomaticDPOnegativeforprompt-conflictedblockedcase. Rootadjudicationreview-v3JSONSHA0964a8e10371d00fd21bcd6337d1c1518aaa3361fa036e110c37d379fb7f5bad.
- Luna2finished75accounted thenrestartedactual1364869 with27 freshdirectory-reducer cases (26MuSiQue1SciFact), native27/currentadmit/conversion/materializer54decisions. Rootcontroller1364868/planSHA7e28c89c911899a96aa789cf2949ed4133a4550ba8c71ec326b0c4054b4d14f3. 300additionalcandidates correctlyheld (Qasperequivalence/sourcepending);quantity256minimum wasremovedfrompreparationso27validcasescouldproceed. Nextsource-reviewedrepairs beingprepared asautomaticbacklog; don'tpadquantity. These27preflightrefs usebarefile.readTextcalls; doNOTpublishpreflightrowsasstaticsuntilvisible-inputproof repaired. Actualmodelteachers choosevisiblefile reads; reviewgeneratedsourceconditioningbeforepublication.
- SpaceBunnynext256finishedALL256accounted; nowidleawaiting300newworkflowpairreducers/newimmutable36(prompt+taskclose+existingemptyresponseclassifier) beingprepared bysourceagent. Reuseofexistingworkflowquestion sources explicit; newcase/shape notnewsourcecoverage. No queuedrolloverclaimeduntilactualrootreviewlaunch.
- FullTypeScriptemit changedcanonicalcompiledsource-review.js formattingd0c→48c439withSAMEunchangedsourceTSf41f4, invalidatingarmedcontrollers. Rootrestoredexactroot-reviewedcompiled d0cfromsealed34; no policy change; Bonsai/Luna1pinsmatchagain. Luna227planupdatedreviewedpolicybytepinonly withexplicitrootreceipt, originalreports retained. FutureagentsmustemitONLYownedfiles, nofullmutablecanonicaldist rewrite.
- BothapprovedDGXstorage trees successfullymoved/verified/path-preservingsymlinked. `bgkit-data-nvme.release-receipt.json` backupremoved13:15:27, exactchecksumsubsetzero differences/fullstrictcomparebeforecutover;internal346GiBfree/external544GiBfree. `.cache` untouched. Firstsdkb-runsrelease11:48 remainsvalid.
- Active50-minutesleepstillpendingwhileworkersuccessor supply/36releaseprepared. Lastsleepended11:19; allurgentfixesbeinglogged, nobackgroundagentwakeupconfigured. Otherrootwork untouched.

---

# Central DGX pool is live — 2026-10-02 12:59 UTC

- Actual DGX user service `natlang-qwen36-pool1024-v3-20261002` is running supervisor727346/collector727374; local sync1334920. Campaign `runs/dgx-qwen36-pool1024-v3-20261002/`, remote same campaign name under~/natlang-remote. At12:58 observed256 running model requests/0waiting,320 case-start events, KV38.38%,27.3GiB available RAM. One realshared256request gate,320 active-case workers,8192MiB Nodeheap,16384context,40turns/384requests percase,8transportretries/5s exponentialbase, defaultwalloff. Separate12GiB disk andMemAvailable preflightguards. This is production utilization evidence; prior captured/synthetic rates are not productionquality/throughput promises.
- Root independently verified all17,660 sealedruntime34r4 file hashes, all630 approved33 base paths preserved with ten changedfilebytes (12overlays2identical), private17,029dependencyfiles and zero symlinks. ManifestSHA9c70c187cc944bac1c625132d3719116fc37152d5a293efda97bd0d56ed7e0fc. Correctbase `runs/qwen-prompt-review-20261002/runtime-v33`; earlierwrongbase481file candidates neverused. Runtime transferred to remote campaign/runtime-v34. Native1024 fullreplay/currentadmit/conversion/materialization passes13,590decisions,0unlinked/0failures; rootchecked allactualrows+detailed rowhashes. Three remotezero-model fixtures(invoice/TreeDST/directory) accepted. ExistingremoteNode18.19.1 actuallyused thoughpackageenginesdeclares>=22.13; rootdeploymentreceipt recordsinterpreterSHA/mismatch. Follow-up supportedisolatedinterpreter deployment, never mutatefrozenruntime.
- New1,024 v3IR SHA22de98554568dd6d4cc503ea3acff3047822247de67132a53c49c05ba7ae653e,300workflowpairreducers29.3%,705workflowprimitives19TreeDST. Native preflight caught malformedescaped return annotations in v2beforelaunch; v3versioned300reducerIDs/schema strings only, source/golds retained. Alloldv2staging remains unlaunched. Sourcefreshness must not countthe same newdraft stagingas priorgeneration; rootconfirmed0case/primitive-source overlap withcompleted2048.
- Pool final-only export keepsatomic percasejobslive; importer sync readsjobs, final aggregateatcompletion/drain. Journalroot namesjournal-pool.jsonl/journal-cases.jsonl matchsyncallowlist. Runnerpublishespausedsetup underownedlock (previousdraft reacquire bugcaughtfixedbeforelaunch), recordsinfrastreaks in completionorder, gracefuladmissiondrain separatefromhardabort. Hoststop withsystemctl --user stop requestsdrain; explicitremote hard-abort.request onlyforreviewedcancellation. Do notcallresource failureswronganswers orDPOnegatives.
- LiveNode18 produced high-concurrencyAbortSignal listener warnings; Luna source agent auditing expected256 listeners vsactualcleanup leak. No runtime mutation/inflight restart. Firstpool resultimport/pressurecheck pending.
- SpaceBunny all8empty-response recoveries producedexactaccounted outputs; new35 actualnext256worker1333346running. Firstrollovercontroller1331595 crashed waitingfor nonexistent initialstatusfile beforelauncherwroteit; rootfixedcanonicalhelper towaitonlywhilethe authoritative launcherPIDis live, otherwisefailneedsreview. Re-reviewedv2plans/guardedrollovercontroller1333341 launchednext256 afterall8complete. Oldfailure/raw8preserved; originalnext4archived248completed+8incomplete transferred, neverclaimed256trainingrows. Genericfinishguards unchanged. Rootreceipts under space-bunny-runtime35-prep and actualcampaigns.
- Bonsai/Luna1 waitingcontrollers stillallpinsvalid afterproviderhelperfix. Luna2repair59of75finished at12:58; qualityagentpreparingnextbacklog urgently before50minutesleep. SpaceBunny next256at~36entryrecent; DGXnextbacklog/recent13rejectionreview delegated. Continueactive50minutesleep/checkloop oncecontinuitysupplied; lastsleepended11:19, urgentdeploymentkeptrootactive.

---

# DGX central pool preparation and completed campaign — 2026-10-02 12:44 UTC

- Completed DGX next2048 campaign has **2,048 imported exact artifacts: 1,937 admitted and 111 rejected** at import time. All128 batches finished with exact accounting; no assignment-held artifacts. These are admission candidates, not published training rows. GPU generation is idle while the next pool is reviewed, not claimed continuously busy.
- Next reviewed selection:1,024 cases,705 fresh workflow primitives(662 invoice43customer),19TreeDST,300 new workflow pair-directory reducers(29.2969%). Reducer pairs reuse600 source questions also included as standalone primitives; new task shapes, not600 additional fresh sources. Qasper generation-held records excluded. Current-policy/native/materialization proof and sealedruntime34 still pending launch.
- New pool runner `scripts/reviewed_single_pool.py` targets320 active cases sharing one actual256-request gate. Final-only aggregate export avoids repeatedly rescanning all rows; atomic per-case jobs remain live/importable. Separate graceful drain stops admission and allows active cases to finish; default pool wall budget disabled. Keep40turn/384request limits per case. Exactsource/runtime/config authorization required. Distinct memory guard is being added; existingmin-free-mib checks disk only. Journal paths must startjournal*.jsonl for remote sync allowlist.
- Initial closedruntime34 identified missing direct provider-request-controls dependency before launch; source agent building separately versioned complete closure from33 with only reviewed overlays, ownphysicalnode_modules, no symlinks. Do not use unsealed shared-dependency34draft or mutate33.
- SpaceBunny next4 is finished at supervisor scope but **8 entries are incomplete**, all exactProvider returned an empty response with zero reply/delta bytes. Transport issue, never DPO negatives or positive examples. Canonicalretry.ts now recognizes that exact SDK error under existing bounded exponential backoff/jitter; typecheck passed. Separate8-case recovery35 and next256 successor being prepared. Direct next4→next5 rollover correctly blocked; authority oldrunning field stale until explicit recovery reconciliation.
- Luna2 automatic handoff succeeded:actualworker1317866 using approved33,75 source-valid repair cases. Luna1 predecessor1159952 still active and256-case33 successorcontroller1308753 armed. Bonsai778887 active with256-case33 successorcontroller1324745 armed; rootplanSHA959077ccb49fe49580094a062f628972b665279cb91ff2de46bd631b8f2c722a, all654pins independently checked. Existing33 parentdependency limitation explicitly retained; newDGX runtime seals ownclosure.
- Storagefirstsdkb-runs backup released with full proof; secondbgkit-data-nvme copy finished and fullchecksum verification running690063 under userunitnatlang-storage-recovery-v2-20261002. Internal201GiBfree/external544GiBfree; nosecondcutover/release claim. `.cache` unchanged.
- Lastactual50-minute sleep ended11:19. No nextsleep while urgentGPU pool/provider recovery prepared. After actuallaunch/armedcontinuity, continue active50-minute sleep/check convention. No independentagent scheduler is configured.

---

# Standalone variants and independent Luna handoffs — 2026-10-02 12:09 UTC

- Five MuSiQue standalone answer forms now root-approved in canonical builder:1,318kilometres(819mi),4April2017,FirstSino-JapaneseWar,GreatKhan,Neutral (actual registry preserves spaces exactly). Receiptmusique-v3-quality-receipt.jsonSHAbf1630fa…d78279a +root-musique-v3-review.json in next2048/rejection-review-1119. Native5standaloneaccepted,4rawverbose rejected,rawfulldateaccepted,5idempotent,primarygoldspreserved. Metadata clearly separates acceptedanswerstring/hash fromcapturedteacherraw/hash. WeakNiger unaccepted; jazheld. Existingraw/static/frozen snapshots notrepublished; builderversionednewIRs only. Catalog refreshstillneededtoinclude reviewed transformations.
- Canonicalstart_reviewed_luna_slots.py now accepts1or2distinctnumberedslots, allowing each tohave its ownreviewed automaticplan. Worker number1/2required because monitor uses it. Handoff replaces onlyboundexistingauthorityslot andstillrequires exactcompletion/PIDabsent/artifactpins, cannotspawnextraworker. No new unit tests; syntaxparsed. QualityagentpreparingactualsingleLuna1/Bonsaiplans now, enlargedLuna2repeat-repairqueue next. Nine-case repairdraftwas too short toguaranteecontinuity; explicitlyreviewedpriorrejectedsource repeats allowed, notclaimedfreshcoverage orautomaticDPOpairs.
- Local33 source snapshot hasno ownnode_modules and resolvesworkspace dependencies;630source manifest doesNOT provefull dependency closure. Existinglocalteachers sharethatlimitation. New34poolmustinclude pinnedactualdependencytree/tsconfig, asremote32alreadydoes. Nevermutateexistingapproved33 toconcealthis.

---

# Exact source holds reviewed — 2026-10-02 12:03 UTC

Ten exact MuSiQue source/question contracts are now held in current canonical/compiled source admission: nine unresolved relation/oracle cases plus the source gold typojaz. Receipt `runs/dgx-qwen36-generation-next2048-20261002-v2/rejection-review-1119/source-hold-receipt.json` SHA84b45bac9c6af6b143344c40f10302419bbb6eaf5322c4c0eae19257bbf5037d; root-source-hold-review.json recordsrootapproval. Recordmatching checks source revision/snapshot/fullprompt/originalgold; ID-only metadata lookup still exposes pending reviews. Alltenmatchingrecordsheld, mismatchedrecordnotheld. Raw/goldsretained; nonebecomepreference negatives. LadyGaga2017 modifierambiguous, notproofofdatecontradiction. Current source-review compiledSHAd0c15955…1fffdd1; sourceSHAf41f4bc7…807743a. Immutable32/33unchanged. Nextselectedqueuesmustrecheckcurrentpolicy; qualityagentpreparingactualautomaticplans.

Five source-backed standalone answer variants are staged separately; four capturedverbose teacheranswers stillformatrejects, one directfull datecanmatchitsreviewedvariant. Pending finalnativeproof/metadata cleanup, notyetpublished. No blanket fuzzy/substr matching. Existing Oklahoma annual-event reviewedcontract preserved; weakNiger candidate not accepted. Staticjazholdpreventsbad misspellinglabelpositive; latercorrectedversion mustpreserveoriginalhistory.

---

# Pool scheduler preparation — 2026-10-02 12:00 UTC

- Current32batch-slot scheduler reserves8request permits per process until an entire16case batch completes. Observed107running/0waiting with28batchesunleased at11:47 confirms underfill is plausible despite256 ceiling; current livework unchanged. Read-only audit recommends one collector pool with one real shared256 KvBudget and320case workers, which can replace individual finished cases immediately. Existing512-case32process ceiling is arithmetic, not a cross-process semaphore.
- Canonicalcollector now serializes/coalesces aggregate exports (100ms window) instead of overlapping full-source merges from every worker. Jobs remain atomic/durable first; final merge is awaited; export error stopsnewcase admission, in-flight cases finish then error surfaces. Native8case concurrent replay completedoutoforder, mergedsourceorder correct, all exactjobs/currentadmit/materialize pass,0modelcalls/unit tests. Receipt `runs/collector-merge-coalescing-review-20261002/audit.json`. Not deployed in immutable runtimes yet. Agent preparing separate reviewed pool runner/explicitdrain hook; don'treinterpretexistingcount/wallbudgets or mutatelive32campaign. Per-case40turn/384request limits remain; poolwallresourcecontrol defaultoff, not new language deadlines.
- Root rejected draft MuSiQue registry that included five full explanatory final responses; they still violate answer-only return contract. Only standalone supported answers can become per-source exact alternates; raw prose staysrejected. `jaz`gold source typo must be held pending versionedcorrection rather than training staticjaz. Registry/sourcepolicy agentsarecorrectingdrafts; current publication unchanged. Local automatic successors await sourcepolicy pin settlement, allteacherslive.
- At11:59sync healthy:1646of2048exactartifacts imported. Nativecase-level admission counts are separate from publication. Storage secondtreecopy stillrunning; firstreceipt reclaimed155GiB internal. Continue50minactualsleep once urgentautomaticcontinuity supplied.

---

# Storage milestone — 2026-10-02 11:49 UTC

- `sdkb-runs` retained backup is now removed after fresh zero-difference checksum subset proof. Durable release receipt remotely `storage-relocation-20261002/sdkb-runs.release-receipt.json`, removed11:48:19UTC, helperSHA4561fd69…ecb5f9, exact destination/symlink verified. Internal available space increased47→202GiB (95→77%used). Original full178.7GB copy remains on external; `.cache` untouched. Second `bgkit-data-nvme` is now copying with original retained, statephasecopying; external679GiBfree at11:49. Do not claim second move complete until exact full comparison/cutover/release receipt.
- Recent local teacher observations: Bonsai66completed queue entries, Luna1 167, Luna2 35, SpaceBunny72; each has recent activity and zero unaccounted terminal finishes. Entries may cover multiple cases; these are not final training admission counts.

---

# Live continuity and rejection review — 2026-10-02 11:47 UTC

- DGX shared generation remains live: dispatcher633031, service `natlang-qwen36-shared-next2048-20261002`, 256 aggregate request ceiling/512 active-case ceiling. These and storage are USER systemd units: use `systemctl --user`, not system scope. Latest pinned audit at11:35:1303imports/1205admitted/98rejected. All98wrong_return, not transport failures. MuSiQue37final-content/13explicit-return traces show no reasoning-channel/parser contamination. Review at `runs/dgx-qwen36-generation-next2048-20261002-v2/rejection-review-1119/`; seven source-pinned answer variants require individual approval, ten source/oracle disputes require holds, no substring matching or unsupported DPO negatives.
- Luna2 actually restarted11:38:PID1278024, fresh187-case queue `runs/luna2-successor-v32-next-20261002/worker-2.queue.jsonl`, native187/317decisions, root plan SHAfc2e212abd81bc958f7fa8d020d9698bfd864914a5df64c00cd18be206627b4d. At11:45 already26terminal-accounted entries. Both Luna workers and Bonsai are live; their older queues/runtime24 remain unchanged until reviewed handoff.
- SpaceBunny next4 startup had failed before requests because source-review hash changed. Root rechecked all256cases/1599decisions against current admission; new reference-review-1128 retained alongside original. Relaunched11:31 launcher1273971/supervisor1273973 with separately versioned provider supervisor fixing large-JSON stdin accounting. Actual observations live at11:45 aroundentry54. Root plan SHAe3a3e8d7e3a10967900734dd0d6fb8d7ce7d9261c5bf19bb38ebab690de4ecfa. Free provider only, no distillation flag, retry/backoff retained.
- Canonical OpenRouter launcher now publishes paused_setup_failure after startup errors ONLY after obtaining its campaign lock, avoiding stale starting status and avoiding duplicate launchers clobbering running status. Shared successor PID liveness now checks actual non-zombie PID, including versioned supervisors/provider launchers; old basename substring falsely reported them absent. PID reuse conservatively waits rather than allowing overlapping teachers. Syntax parsed; no unit tests run.
- Frozen33 is a minimal prompt-only candidate derived from32: manifest27cda05b881202d17c87042698c4764d056312d9a7980719d6b75d7ba0336f9e, exactlysrc/native/prompt.ts+compiled counterpart changed, other628files identical. Generic typed-string JSON-return guidance; no parser/gold changes. Exact48TATQA hold remains in current admission; not included in frozen33. Next queues must exclude current pending sources using current admission. Frozen33 is NOT yet deployed; current teachers unchanged. Native187-case supplement passes317decisions. Agent preparing actual Bonsai/two independent Luna slot controller plans, SpaceBunny and DGX next backlog.
- Storage corrected full checksum still runningPID637976/user unitnatlang-storage-recovery-v2-20261002 at11:45; no deletion receipts/space reclaimed yet. Internal47GiBfree; external690GiB. `.cache` untouched.
- Continue actual50-minute sleep/check loop after next automatic queues approved and armed. Previous sleep ended11:19; no new sleep yet while urgent continuity/review work proceeds. Background monitor does not wake chat.

---

# Hourly check — 2026-10-02 11:25 UTC

- Actual50-minute active sleep completed; check performed11:19UTC. DGX592 remainder finished592unique artifacts,573admitted/19rejected. Automatic2048handoff succeeded; controller1245179recordrunning, sync1253498, remote dispatcher633031/service `natlang-qwen36-shared-next2048-20261002`. The first two clean batches ramped32collector slots×8requests to256 ceiling at10:54. Current512 active-case ceiling;11:23 has56clean batches/0failed,88of128batchesleased. At11:19:1020saved/927admitted/93rejected, actual130runningrequests/0waiting, KV23.9%,21GiB available,624MiBswap. Raw counts and decisions are not publication claims. Luna rejection review is investigating new93.
- Runtime remainsv32; canonical JSON-return prompt improvement is awaiting immutablev33. Exact48-million sourcehold active in currentcompiled admission; frozen32 untouched.
- Storage recovery failed10:53 BEFORE deletion because it specified itemization twice (`-i` plus `--itemize-changes`), which makes rsync print unchanged files. Backup root directory mtime also changed during earlier partial deletion. Corrected helper uses ONE itemization and ignores only directory mtimes for a retained-backup subset check, retaining checksum/file/link tests; second-tree full comparison remains strict. Metadata-only corrected check was empty; fresh full checksum is running under `natlang-storage-recovery-v2-20261002`,PID637976 since11:21. No reclaimed-space claim; two original/external copies still retained. `.cache` unchanged.
- Local continuity check: Bonsai778887 andLuna1 1159952alive. Luna2finishedall256cases around10:02, authoritywasstale; Luna agent preparing its successor and inspecting SpaceBunny next4starting status. Do not describe two active Luna workers until restarted and verified.
- Next immediate work: restart suppliedLuna2/armitsnextqueue, frozen33minimalprompt+exactsourcehold review, complete93rejectaudit, fixSpaceBunnyifneeded, verifydisk recoveryreceipts and keepDGXnextbacklogprepared. Continue50-minute active sleep/check convention once urgentcontinuity work is supplied.

---

# Current handover — 2026-10-02 10:26 UTC

- **DGX throughput reconfiguration is deployed.** Server `natlang-qwen36-nvfp4-server` retains pinned Qwen NVFP4 revision and compatible `eugr` ARM64 vLLM image; now max sequences256, GPU memory utilization0.65, context65536, FP8 KV, Marlin MoE, prefix caching/chunked prefill and8192 batch tokens. Explicit `/usr/local/bin/vllm` entrypoint is required. Initial restart omitted that override and failed before model initialization; failed container retained, corrected server healthy since09:37. Old16-slot container is retained stopped for rollback.
- **Generation initially resumed09:57:39 (superseded by recovery below):** `natlang-qwen36-shared-generation-20261002.service`, dispatcher591048, local sync1229162. New campaign `runs/dgx-qwen36-generation-successor1024-high-concurrency-20261002/`; shared37-entry queue transfers592 never-started cases. Original1024 assignment safely drained:432 exact results,408admitted/24rejected, no live old collectors. Old authority entry archived as drained/remainder-transferred, not falsely completed1024. All raw data preserved.
- **New parallelism:** eight16-case collector batches expose128 requests initially; after two clean complete-accounting batches, sixteen batches expose256. At09:59 actual128 running requests, zero waiting, KV19.6%, host25GiB available. Dispatcher leases whole original batches to whichever slot is free; seeds/indexes/keys/golds/per-case budgets retained. New paths prevent mixing old/new exports. SIGTERM stops new leases and drains children; systemd KillMode=process, TimeoutStopSec=infinity, Restart=no. Three consecutive collection failures pause new leases; provider cooldowns block new leases even with active children; nonzero children/missing terminal events pause. Clean accounting is required to advance the ramp. No language-level timeout was introduced.
- **Production recovery10:05:** first completion crashed dispatcher on incorrectly nested queue-key lookup; preserved original crash logs and38 saved results. A separate supervisor accounting defect read large JSON from stdin after importing Node modules, got EAGAIN and mislabeled all16 completed SciFact results invalid IR. Fixed digest input to a temporary regular file, revalidated exact16 saved/exported rows without model runs. Deployed distinct dispatcher-v2.py/run_queue-v2.py, pins-v2/handoff-v2/root-approved-launch-v2 receipts. Actual service `natlang-qwen36-shared-generation-v2-20261002.service`, dispatcher592409; same sync1229162. No old children remained at restart. Batch15 terminal failure stays historical; root recovery receipt establishes complete accounting. Other saved jobs resume with collector provenance checks. Canonical supervisor also classifies digest-helper failures accounting_error; canonical dispatcher now writes paused status on exception (that extra status catch is for the next deployment; active v2 keeps earlier loaded code).
- **Exact deployment/import distinction:** remote `bundle/cases.ir.jsonl` is full original1024 execution IR SHA314f3026…b255d; local campaign `bundle/cases.ir.jsonl` is exact592 import allowlist SHA208efd6f…fbda5f; local `bundle/execution-source-1024.ir.jsonl` retains full input. Active queue SHAe66c7978…01b75. Agent later corrected its separate local candidate package to a different source filename/queue; DO NOT overwrite active remote from that newer proposal. Root remote pins SHA`d44f5cf998e7e98b652edafbacea7151e7046f00002ecb3fbe0a3b1afdc5e38d` include all actual runtime files/dependencies plus630-file frozen manifest; actual copied tree includes node_modules and tsconfig. Root receipt/launch copied into active local campaign. Frozenv32 finish:true manifest remains472230b7…454e3. All slot/dispatcher journals live at remote campaign root for existing sync.
- **Measured benchmark:** `runs/dgx-throughput-20261002/results-upgraded/`, client under `benchmarks/dgx-qwen36-throughput-20261002/`. All32/64/128/256 cohorts finished with zero request failures/OOM signals. Synthetic1500-in/400-out throughput444.5/540.3/634.8/699.2 outputtok/s; captured teacher-context throughput336.9/458.6/557.1/654.5. Captured256 TTFTp95≈21.17s, response latencyp95≈148.64s. Old16 synthetic baseline343.0tok/s. Synthetic outputs were capped reasoning-only; captured benchmark cycles four repeated contexts, benefiting from prefix reuse. These are transport/scheduling measurements, not correctness or diverse full-trajectory speed. Do not claim Reddit's2835tok/s was reproduced. Check actual admitted cases/hour and production rejections next.
- **Backlog STAGED/ARMED, not yet running:** `runs/dgx-qwen36-generation-next2048-20261002-v2/`,2048cases/538reducers26.27%/1510primitives,12035approved reference decisions,0unlinked. Current admission/materializer/nativev32 all2048passed. Remote bundle staged and repinned with corrected supervisor and canonical dispatcher exception-status handler; runtime32 reused. Automatic successor controller1245179 waiting; root plan SHA`1a717440be62d174a5b0d3780cd1371e7da37f3ae1c807af47634dd4a6175d70`, remote pins SHA3466484b…64f8c3. Starts only after all592 current artifacts imported/unique/no assignment holds,36 clean terminal batches plus independently revalidated16-job recovery receipt, inactive old service and exact remote accounting of all37 batches. New32 slots×16cases expose512 active cases with8 requests/slot:128→256 request ramp. Root journals preserve sync compatibility. Source groups may repeat within TRAIN; exact prior Qwen case/source IDs excluded. Proposed v1 package superseded, never launched. Local Bonsai/two Luna/Space Bunny continue; proposed Bonsai/Luna1v32 rollover still needs root controller approval.
- **Storage correction:** sdkb-runs full checksum passed and symlink cutover completed09:05. Backup deletion failed on root-owned directories; backup remains, internal space not yet reclaimed, secondbgkit-data-nvme move not begun. Recovery helper `scripts/recover_reviewed_dgx_data_relocation.py` SHAaaf29694…ea4850 is now running as `natlang-storage-recovery-20261002.service`,PID594985, lowCPU/IO priority plus2GiB MemoryHigh. Fresh subset checksum precedes exact-backup-only Docker deletion; nested-mount guards on host/container. Receipts under remote storage-relocation-20261002. Not yet complete. `.cache` stays. Old training containers/heartbeat remain stopped. Do not report storage verification as still running.
- **Verified10:09:** ramp reached256 requests at10:07 after two clean completions. Three batches now have complete accounting, zero new batch failures since recovery. New campaign187 imported artifacts; latest27 all admitted. Rejections are under Luna review.
- **Rejection improvements:** audit first12rejects found six correct saved answers with malformed final returns. Canonical `src/native/prompt.ts` now adds generic savedJSON/string-return guidance; NOT deployed in frozen32, stage in future33. Exact TATQA source13d32af1-1901-4340-b9b8-023ed52ec0ec now centrally held in canonical and current compiled source-review policy because supplied142/94 table has no unit header while gold48/million conflicts with evidence-scale contract. Gold unchanged; no preference negative. Current admission helper verifies source_review_pending; historical raw/provenance retained. Full audit in current campaign/rejection-review-v1.
- **Verified10:26:** current transferred592 campaign425saved/408admitted/17rejected; with original432,857/1024generated. Automatic2048controller waiting, production/storage services active.
- **Next:** inspect production throughput/admission, deploy reviewed2048 successor before queue runs dry; complete approved storage recovery and localv32 rollover; review unpublished causal Qasper static candidate. Continue requested50-minute active sleep/check loop after immediate deployment follow-up; filesystem monitor does not wake chat.

---

# Current handover — 2026-10-02 09:06 UTC

Read this summary before the historical entries below. Process IDs and counts in older entries are historical; current queue ownership is in `runs/generation-check-20260930-hourly/check.json`.

- **All DGX generation uses Qwen NVFP4**, `nvidia/Qwen3.6-35B-A3B-NVFP4` revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`, endpoint `127.0.0.1:8082`, server `natlang-qwen36-nvfp4-server`. Sixteen requests across four collectors. Horizon services are stopped and disabled; its raw results and checkpoints remain archived. All 257 queued Horizon cases have exact IR coverage in fresh Qwen attempts. FP8 download was cancelled; Gemma NVFP4 has finished downloading and verifying but is not deployed.
- **Current DGX queue:** `runs/dgx-qwen36-generation-successor1024-20261002/`, 1,024 cases (256 reducers / 768 workflow), 16 aggregate requests. Remote service `natlang-qwen36-generation-successor1024-20261002.service`, bootstrap546911/supervisors546934–546937; sync1184033. Runtime v32 is ACTUALLY running with current eval.finish:true schema+implementation; manifest `472230b75e4bf411a88c958e45e4b419f232dfaa4d58392840d006dd002454e3`,630files/fourchanged. Native1024/5591decisions, zero unlinked. Prompt-only31 was an undeployed superseded draft.
- **Runtime deployment receipts:** successor `root-review.json`, `root-approved-rollover-plan-v3.json` (SHA20cbe3ffda3d79030dbdf933164b6f44d8786a096c359e775ccbf7ec7c04e256). Old512 finished with511rawresults plus one explicitly retained malformed-tool partial; all terminal journals reviewed. Fixed controller recognition of absent completed transient unit (systemctl exit4). Initial successor startup failed before any model request because queue source path expected root/cases.ir.jsonl while only bundle copy was staged. Copied identical pinned IR to root, added exact source pin, archived source-layout failure status/log, restarted approved bootstrap and sync. Current status running. Never mutate old frozen snapshots; new freeze guard refuses advertised finish:true without compiled schema/dispatch.
- **Completed Qwen batches:** pilot 55 admitted / 9 rejected of 64; mixed batch 222 / 34 of 256. All rejects have source/trajectory audit reports in their campaign directories. No edit-runtime or parser defect was established. Wrong edit strings, printed result-tool calls, final-answer prose, and real arithmetic/semantic errors account for many failures. Apollo/LaHave source disputes are held in current source-review policy; numeric golds remain exact.
- **Local generation continues:** Bonsai, two Luna slots and Space Bunny. Luna slot2 has already handed off to v32 as worker1177972; Space Bunny next4 v32 controller1177984 is armed. Bonsai/Luna1 v32 successors are being prepared. Luna slot 1 successor is running as worker 1159952 after reviewed handoff v2. Original controller stopped safely on one wall-budget partial; its raw attempt is retained and is neither a positive nor a timeout-only DPO negative; see `runs/luna-slot1-spacebunny3-successor-20261002/`. Space Bunny currently uses `runs/space-bunny-workflow-next3-20261002/`, one request, free provider only, no distillation flag. Preserve existing retry backoff.
- **IR preparation:** `scripts/prepare_generation_ir.mjs` now carries the existing reviewed TATQA and MuSiQue contracts. All 1,212 modern cases are preserved; 17 MuSiQue changes passed native checks, and a second preparation pass is byte-identical. Receipts: `runs/generation-check-20261002/manual-0734/generation-ir-reviewed/`. Third source-pinned TATQA proportion display fix is prepared and native-verified. Full 1,212-row fresh replay has 1,210 static-positive candidates, two source quarantines, and separately 910 generation-eligible cases. Qasper’s 300 generation-held static candidates have a confirmed visible-evidence gap (paper reads did not print content); repair is in progress before publication. Publication/catalog/default recipe refresh is still pending. Distinguish generation/failure-label holds from static-positive eligibility; do not silently discard Qasper or other archived corpora.
- **Storage:** `.cache` stays in place. `sdkb-runs` copy is checksum verifying again; `bgkit-data-nvme` moves next. First verification paused safely when an obsolete hourly training heartbeat wrote new logs. Its exact cron entry was backed up and disabled; the incremental copy resumed at 08:36 under PID 510704. Unit `natlang-storage-relocation-20261002.service`, status under `~/natlang-remote/storage-relocation-20261002/`. Originals are removed only after verified external copies and path-preserving symlinks. Do not restart the old training containers.
- **Check-ins:** user wants an active 50-minute sleep/check loop. The background monitor audits files but does not wake this chat. Agent mailbox messages can interrupt sleep; keep the absolute deadline. Use `followup_task` to restart a completed subagent: `send_message` alone does not do so. This mistake delayed the DGX queue refill after the mixed batch; corrected by explicitly restarting the preparation agent and launching the next queue.
- The user withdrew the no-backward-compatibility/no-state-migration instruction as belonging to another project. No such restriction applies here. Preserve archived training data and use reviewed transformations.

Respect unrelated concurrent workspace changes, especially program-improvement code and `training/data_sources.json`. Stage only owned edits. Training has not started; the final dataset still needs publication, rendering, deduplication/leakage checks and the 25% reducer mix gate.

---

## Missed agent check-ins: mechanical audits did not wake the agent — 2026-10-01 20:29 UTC

User asked why the hourly check-in did not fire. Monitor584649 actually persisted audits at18:09,19:10,and20:10UTC and remained healthy at20:27. `monitor_generation.py` only writes files/stdout; no agent wakeup or chat notification is configured. Root ended its active turn after the watchdog change, so independent investigation and user reporting did not run hourly. This is a follow-through failure, not a stopped generation monitor. Do not describe mechanical audit completion as agent investigation/check-in completion. Do not promise a future chat check-in without an active turn or a verified scheduled-agent delivery mechanism.

Current20:27 heartbeat: Bonsai live; Luna783977(runtime24 successor) and483515(old predecessor) live; controllers have0alerts. Last20:10 completed Bonsai spool20exactresults/20admitted/120decisions. Saved reports18:09 through20:10 still need agent rejection review and the live runtime24 stream/supervisor handoff needs inspection. Current aggregate report scope changed at queue handoff; do not interpret falling current-queue totals as loss of historical training data.

---

## Observation-aware collection supervisor approved — 2026-10-01 17:38 UTC

User requested implementation after rejecting blanket timeouts. `scripts/run_bonsai_queue.py` now defaults to **no silence cutoff**. Optional `--no-observation-seconds` is a collection resource ceiling, reports `no_observation_limit` with stuckness unknown, and never determines answer correctness or DPO eligibility. This is an intentional status/policy change from the unconditional 300-second `inactivity_timeout`. Explicit case wall budgets remain 1200 seconds for Luna and 4800 seconds for Bonsai; limits do not enter language semantics.

A bounded incremental reader starts at each child's log EOF and reads only validated content-free provider phase/stream events. Increasing nonempty delta-event and byte counters count as actual output; cumulative counters, log polling and phase starts do not. Delayed stream-only observations retain their event time. Increasing saved replies and local decode counts are separate signals. Explicit retry waits are reported separately, exclude the optional silence budget, and do not reset the output clock or wall budget. Observations are per job, not proof that every concurrent request is healthy or that the model is making semantic progress. Final telemetry is read before finish accounting. Raw partials and first terminal outcome are preserved.

Supervisor SHA **ddbb7dde6c0b2bda8e305f803a70e464d4643d2513598c8c0e9b1689c9868b5a**. Independent read-only source review receipt `watchdog-observation-v1/agent-source-review.json`; root policy and syntax review `root-policy-review.json`; launch receipt and actual zero-alert controller check `launch-receipt.json`, `controller-health.json` in `runs/generation-check-20261001-resume/watchdog-observation-v1/`. Syntax parsed; no tests or model calls for this change. Runtime24 remains immutable; its real stream producer has not yet run in these queues.

Only waiting controllers were replaced: Luna **740896**, plan v10 SHA **301c56e69f65c4091e4e7ba922f95669a98783113c448b976c64c996e64a76fd**; Bonsai **740897**, plan v4 SHA **361b66770c58fa2b24c3a511931b90c2235964f76cac52baf4840814631bec38**. All six exact storage recovery approvals are now `queue-N-approved-watchdog-v1.json`, with unchanged queue/runtime/command bindings and the new supervisor hash. Current teachers **392661/483514/483515** continue under their old already-loaded supervisor code, including its old watchdog. **The improved policy applies at the next campaign or recovery restart; it is not yet active in those running processes.** The old supervisor is preserved byte-for-byte under `watchdog-observation-v1/legacy-supervisor/`. Do not interrupt in-flight work merely to claim immediate deployment. Monitor584649 and server watcher368408 continue.

Actual17:09:26 hourly sweep: 0 new non-success finishes, 0 blocking reader errors. Main1071saved/1031admitted/40rejected/8712approved decisions; vs16:09, +139saved/+134admitted/+5rejected/+1160decisions. Completed Bonsai spool32exact rows/29admitted/151approved decisions, 0missing/source errors; its3rejections include rows already visible in the previous main scope. Exact five-new-ID audit is delegated; do not infer teacher attribution from completed-only spool. Next full check18:09:26UTC.

Metadata correction: preference subset audit's transcribed recipe31 digest was wrong; the agent rehashed the actual recipe and corrected the receipt with explicit old/new evidence. Current subset audit SHA **6969f8da99677a2e96023031969f2ddf320209c5599fc16805713690bb3385f9** supersedes c2b1d309…d297c. Three replay mismatches differ only in model-visible local preview suffixes; console output is exact, but context equivalence is not established. Keep those three held, not automatically normalized or promoted.

Dependency reproducibility audit found frozen24 shares mutable workspace node_modules and its old lockfile omits Pi SDK. No dependency install/change has occurred. Separate immutable dependency closure is a follow-up; source/runtime hash proof alone does not prove dependency identity. Audit `runtime-dependency-reproducibility-audit/` under the generation-check directory.

---

## Timeout policy and preference-pair audit — 2026-10-01 16:47 UTC

User remains skeptical of blanket timeouts. Request/action-cycle deadlines remain opt-in and disabled in future plans; they are collection resource controls, never language semantics. Existing supervisor still kills after300s without a saved reply/local decode progress (explicit provider retry waits exempt), and hard case budgets remain1200s Luna/4800s Bonsai. Neither proves semantic failure. Runtime24 observation does not refresh this watchdog. Investigate an evidence-based policy before changing it; distinguish real stream output, deliberate waits, and silent/unknown requests. Preserve incomplete raw attempts and forbid timeout-only DPO negatives.

Read-only recipe31 preference subset rebuild recovered18causal pairs (15failed_action/3wrong_result) from21exact covered historical parents;3replay-observation mismatches remain held under investigation. All18 source/chosen digests matched current native materialization; no reserved source groups. Full recipe stage has not run because its upstream run outputs do not exist. Receipt `recipe-v31-review/rebuild-current-pairs-v1/subset-rebuild-audit.json` SHA c2b1d30954a6c37ef27abf6a3e79b75e08196a1fae512fe35a64af49cf7d297c. These are candidates pending root/student rendering and final gates, not final DPO training data. Historical43pairs and1632unreviewed failure candidates unchanged.

Taxonomy audit: static6138cases/32652approved decisions include7574reducer decisions (23.20%). Combined pre-render candidates19474/64384 (30.25%) is not the final mix gate. The125legacy program-editing turns must not be counted as reducer decisions merely because the edited program has a directory root. Separate source topic from the operation the learner performs. Receipt `recipe-v31-review/recipe-v31-taxonomy-audit.json` SHA c76939d9f108e7c5f03379ffca602372dffe0035c2b981169274aa1443d694a1. No classification code or canonical data changed.

---

## Runtime24 progress observation and recipe31 approved — 2026-10-01 16:23 UTC

Root verified all621frozen24 hashes, exactly10owned changes from23: contracts type/declaration, managed-session optional progress callback source/JS/declaration, Pi streaming aggregate source/JS/declaration, collector source/JS. No other611files changed; prompts, deadlines and watchdog policy unchanged. Manifest **a0c734ec088378ac5d8378db33b93e7a002083faeafd105b662d72cca2aa8a55**. Real nonempty text/thinking/tool-call delta events/UTF8bytes are counted without logging content. First actual delta then at most once/15s with new bytes; final counters/status emitted. SDK full event iteration plus authoritative final result retains reply/usage; original cancellation signal forwarded, observer failures isolated. Pi provenance `pi-stream-observation/1`, watchdog_refresh:false. This observes transport output, not proof of semantic progress. No default request/cycle timers.

Root again replayed1054native gold references:541Luna/5862decisions,512freshBonsai/2948,exact48repair/11;0reject/unlinked/modelcalls. Reference receipt `root-runtime24-reference-review.json` SHA **1fba0c376064f7bec0a696459ed6b9ac77ea9d593f757017003697d7abf481b2**. Stream provider path itself has not yet run live; inspect first actual24Luna logs/captures when slot hands off.

Only waiting23controllers671283/671284 stopped after nochildren/launch/argv checks. New **Luna701232**, planv9 SHA **b49c2a565aaa7e7e0f8b6c2e7588a301caa7b0c81e46096546d69365138086a5**; **Bonsai701233**, planv3 SHA **1132cd7cb0f2ba669ecc7ac00b14cd0c362dc77fb4a281b75330f4f11e2cdf45**. Future recovery approvals4/5/6 updated24; active1/2/3still20. Actual checks0alerts. Current teachers392661/483514/483515 unchanged20; monitor584649/serverwatcher368408continue. All20–24snapshots andoldplans immutable. Launch/preparation/health under `runs/generation-check-20261001-resume/runtime24-*`. Nextactualhourly **17:09:05UTC**, rootactive.

**Recipe31 default-input inventory draft root reviewed/pointer updated**: SHA **1a85cb393fd5f0d45755ec333a310b865020b76d964b10c335a576ba82372c99**,4037selectedgeneratedtrajectories/3125programs asof15:58:13UTC; allfourstatic6138cases/32652decisions,7reviewedlegacy lanes289turns. Root independently recomputed9requiredresolutions/41historicalpaths(40replacement/1explicitapprovedhold),2laneidentitystagesconsumed, no overrides/blockers. Independently streamed ALL4037positives throughcurrentadmitRow and allcurrent sourceReviewReason predicates:0pending/sourceheld positives;gzip/contenthashes exact.1632failurecandidates allDPO-negative-ineligible. Do not present partial11-IDagent subset as full source-hold policy. Rootreceipts `runs/data-lineage-20260930/recipe-v31-review/root-{inventory,positive-policy}-review.json` and `root-review.json`.

Draftonly: finalstudentrenderer/token/split/leakage/dedup/25%reducermixpending;350Mbuilderdefaultnotstudentselection. No tests/training/GPUeval. Two oldunlinkedrows cannotbe deterministically joined; exactaudit `generated-unlinked-row-linkage-audit-v1.json` SHA5aec6a72…cf0689. Keeprawrows/othermaterializableattempts; no inventedcallownership orDPOnegative.

---

## Hourly sweep16:09 completed; generation continues — 2026-10-01 16:11 UTC

Health `hourly-health-next/health-2026-10-01T160905064Z.json`:0blocking metadata/outcome errors,502canonical events/97finishes/**0new non-success finishes**. Main declared aggregate scope932saved rows/897admitted/35reject/7552approved decision IDs; delta+111rows/+109admitted/+2reject/+1046decisions vs15:08. This scope is not the whole training corpus. Canonical Bonsai spool supplement finds4new completed batches/16exact results,**16admitted/73approved decisions**,0missing/source errors; the main reader also sees still-active batch184, so its two new rejects were not both Luna: one Bonsai wrapper error in batch184 and one Luna invoice-NOT polarity error. Completed-batch spool and main-reader scopes differ.

GPU98%, Bonsai container2.249GiB/6GiB,56GiBfree. Monitor5846490alerts; teachers392661/483514/483515 and future23controllers671283/671284 remain live. **Next full17:09:05UTC /19:09 Berlin**; root remains active. Recipe31 default-input draft4037selected generated trajectories, fresh carry-forward audit pending root review; do not declare student/training readiness.

Separate future24 source candidate exposes real Pi text/thinking/tool-call delta counts and timestamps without content. Uses SDK stream APIs through terminal output plus authoritative final result; observer errors isolated, original abortsignal forwarded. No fake heartbeat, deadline defaults or watchdog refresh. Phase/progress provenance explicit. Source-only/typecheck passes, not frozen/deployed; current23 and oldruntimes immutable. Nextrootreview required before adoption.

---

## Runtime23 approved for next queues; deadlines disabled — 2026-10-01 15:58 UTC

Root independently verified frozen23 manifest **0bdc4ce342cea9ef172e29e0bd8ceceaf223d84add192d95433b7f8dd46ddd03**: all618frozen22 files accounted for, exactly8owned existing source/dist/declaration changes and3new provider helper files;621hashes match. Unrelated fixture/compiler/program-improver changes excluded. Found inherited22 prompt source lacked three optional APPROACH bullets present in compiled JS;23 synchronizes source to existing compiled optional text, preserving compiled TOOLS/FUNCTION/APPROACH strings. Always-used minimal guidance is now actually appended to default collector prompt and removes helper advice at the NL depth limit.

**No request/action-cycle deadline is enabled by default or in either future plan.** They are opt-in collector resource controls, not language semantics. Phase logging and cancellation forwarding are active for Pi collection in23; configured deadline errors remain terminal with first-error latch/partial retention.15s bounded session cleanup is separately identified and best effort; noncooperative SDK work may retain OS handles, supervisor remains process backstop. Stream-progress exposure is a separate read-only proposal, not implemented.

Root scripted native reference replay under the new actual default prompt: **541Luna/5862approved decisions,512freshBonsai/2948,exactoriginal48repair/11;1054reference replays,0reject/unlinked,0model calls**. Source history/golds unchanged; no reference data published as model generation. Receipt `runs/generation-check-20261001-resume/root-runtime23-reference-review.json` SHA **891e0c662495c91286bcdbd1f3826116d7d147587ad052fff55362c59658a4a1**.

Stopped ONLY waiting controllers617447/580812 after exact argv, no successor workers/journals and launch state checks. New **Luna671283** planv8 SHA **23bf8c180777e4a20181d0e64fea7f6e64292ae4174f275624a48177249d0312**; **Bonsai671284** planv2 SHA **14f6cd2288134d5c0b6156862177874dcee08affb05b12c3ea1097224efa28d5**. Both wait; five exact Luna failures and one exact Bonsai storage failure exclusions unchanged. New future storage approvals4/5/6 are23; active approvals1/2/3stay20. All621runtime files/source/reference receipts pinned. Registry updated under authority lock, real controller checks0alerts. Launch/preparation/health receipts under `runs/generation-check-20261001-resume/runtime23-*`.

**Current teachers unchanged: Bonsai392661,Luna483514/483515 remain20.** Monitor584649 and server watcher368408 continue. Next full investigation16:08:45UTC; root remains active. Recipe31 default-input draft is being prepared with lane identity stages, no training/model/GPU run.

---

## Legacy lane identity collision fixed in derived training view — 2026-10-01 15:50 UTC

Root independently verified125rows across approved folder-revisions-v4(66) and folder-api-v2(59) replacements. They share25original turn IDs but different model-visible contexts. Downstream rehearsal deduplicates IDs, so original IDs could silently suppress one lane. New `scripts/namespace_reviewed_turn_identities.py` derives lane/source-file/review-manifest/original-ID/raw-row identities; original row payload and lineage remain unchanged except derived ID and explicit metadata. Root round-trip/hash receipt: `runs/data-lineage-20260930/reviewed-lane-identities-proposal-v1/root-identity-review.json`. Both source files and approval manifests remain immutable.

Pipeline adds a derived stage only for these exact included approved lanes, before normalization/rehearsal. Resolver pins apply even to direct approved inputs. All evidence/output paths are checked before mutation; output run dirs can be outside repository. No full pipeline, tests or training run. Existing unrelated improvement lane changes preserved.

Recipe30 remains immutable draft. Two older generated snapshot trajectories lack complete invocation linkage: materializer emits zero turns, retains raw rows; separate materializable attempts exist, so no program-level drop. Invocation IDs already present in modern collector; do not invent missing historical ownership. Pre-render decision mix30.70% includes source overlaps and is not a final deduplicated25% proof.

---

## User correction: collection deadlines must be opt-in — 2026-10-01 15:42 UTC

User asked whether proposed deadlines are language semantics or training collection controls and expressed skepticism about blanket timeouts. They are collector-only, never language semantics. **Do not deploy the proposed default180s request/250s cycle limits.** Source patch is being revised to opt-in configurable deadlines. Preserve phase telemetry, proper cancellation forwarding and fatal-deadline evidence handling when explicitly configured. No updated source/runtime has been deployed; runtime23 staging is provisional and must be refreshed after this revision.

Existing supervisor300s no-checkpoint watchdog is also unable to prove a provider is stuck rather than productive. Do not justify a shorter universal cutoff from partial-only observed timing: those intervals include semaphore waiting and planning/action processing, not isolated provider latency. Continue investigating stalls with phase evidence; retain incomplete attempts without semantic rejection or DPO-negative promotion. Current teachers continue unchanged.

---

## Provider stall investigation and deadline review — 2026-10-01 15:33 UTC

The Luna293 incident is not proven infrastructure failure. Exact saved-prefix replay matches all 17 requests and reaches request18 in354ms; historical logs omit the pending provider phase. Confirmed collector defects are missing provider cancellation/deadlines and optional guidance that these queues never enabled.

Root chose **180s individual Pi request, 250s shared preparation/planning/action cycle, and 15s bounded cleanup**. A pair of separate180s deadlines could exceed the300s supervisor inactivity window, so the shared cycle is necessary. No fake progress heartbeat or Python supervisor change. Deadline errors must bypass planning fallback and implicit whole-case transport retry. Native eval can convert child errors into ordinary tool failures, so collector must retain the first fatal provider deadline and reject before admitting a row or deleting its partial, even if a parent handles that child failure.

These changes are still source/staging work, not deployed. Quality agent's initial runtime23 clone must be refreshed after final cycle/latch source freeze and independently reviewed. Keep20/21/22 immutable. Current teachers392661/483514/483515 and waiting controllers617447/580812 continue unchanged; monitor584649 reports no alerts at15:32. Next actual hourly investigation16:08:45UTC.

---

## 15:08 sweep; Luna stall and masked prompt bug — 2026-10-01 15:12 UTC

Fullsweep15:08:45.534UTC; next **16:08:45UTC /18:08 Berlin**.0blockingreadererrors, main821savedaggregaterows/788admitted/33reject/6506approveddecisions (notallnewcases). Bonsaidelta9parentbatches/36exactresults:35admitted,1reject,158approveddecisions,0missing/sourceerrors. GPU100%,container2.94GiB/6GiB,60GiBfree. Canonicalevents106finishes/1newnon-success: Lunacase293below. Agentreviewingnewrejections now. Rootremainsactive.

Monitor caught Lunaw1case293 failure14:55:05beforehourlysweep: keyddcb4f2…b905,canonicalfinishSHA `8ea92b2f8577ae9325e311448d4c65e54507ca37d4647f87ed8a935fc22e1f8a`,statusinactivity_timeout460.5sec,17savedresponses/2652tokens/0outputrows. Lastresponse14:49:31,lastrequestedaction console.log(contract.relevant_excerpts). **last_tool_observation belongs to preceding request**, so savedtrace doesnotprove lastactionhistoricallycompleted. Rootthen diagnosticallyreplayed all17savedresponses underSAMEruntime20/seed962/config: **17/17requestSHAmatches,354ms, reached18threquestwithlasttooloutput**;0modelcalls,noanswerinvented,no rawpartialchange. Artifact `runs/generation-check-20261001-resume/replay-luna-293.json`. Stronglypoints toprovider/planningrequeststall, but historicalphaseunrecorded. Do NOT stateproveninfrastructurefailureorsourcerejection.

Rootpreservedpartial/emptyoutput, reviewedexactfailedfinish as excludedincompleteattempt only; noDPOlabel/goldchange/automaticretry. NewLunarollingcontroller **617447** replaces554491aftercheckingoldwaitingworkers[]; planv7SHA `4519dbbf487fa8301e0c5c99d3ad2b4dd069f053b056095fd7f959385855b173`,fivetotalexactfailedfinishhashes; runningworkersunchanged. Receipt `root-luna-293-failure-review.json`,schedulerlaunch-v7. Actualmonitorchecks0alerts; monitor584649/Bonsaicontroller580812stilllive.

**Concretebugs found:** Pi send wrapper omitsjobAbortSignalandhasnorequestdeadline; onlysupervisor300sec inactivitykill bounds it. Source-onlyproviderdeadline/phaseimplementation inreview (no dist/frozenchanges): needscontrolledtypeddeadlinewinnerrace,boundedprepare+cleanup,preciseprovenance,terminaldeadlinebypassesimplicitwhole-case retries. Existingcollector retriesloadpartial responses byexactrequestSHAthenreexecuteactions; no genericrollback-safetyclaim. Futureimmutable23requiredafterrootreview; current20/21/22unchanged.

**Correction of earlierguidance claim:** defaultSystemPrompt=TOOLS_PROMPT; APPPROACH_PROMPT is appendedonlywith--approach-guide, which supervisorneverpasses. Thus threeguidancebulletsaddedinfrozen21/22arepresentbutNOTactuallyusedbythesequeues. Agentpreparingdedicatedalways-usedGUIDANCEexport/defaultteacherprompt wiring, respectingdepthlimitwithoutunrelatedTOOLS_PROMPTdirtychanges orblindfullguideenablement. Do notclaimfuture22workersalreadyreceivethosebullets.

Migrationauditfinds1110unique pendingartifacts/1535reasoninstances;402inline_delegation_ban.129single-obsolete-reasonaccepted/train/teacherartifactsareboundedreplay-prioritycohort, notdata-onlyrecoverable.113missing_observationinstancesacross52artifacts:0exactmarkersinsavedtoolcontent,no safeinvention/pro-motion. Audit `runs/data-lineage-20260930/migration-backlog-audit-v1/`. Nohistoryrewrites,promotions,tests,newmodelcallsorGPUevaluation.

---

## Default recipe30 root reviewed — 2026-10-01 14:50 UTC

Added exactly three static workflowquality3→quality4 replacement edges (IR/results/turns) to catalog; initialpublicationentry/defaultpointer alone did not resolve all historical required carry-forward paths. Existing API/history mappings preserved. Strict legacy reviewed-turn resolver remains unchanged: these static transformed cases use publication/source-equivalence lineage, not a false retained-subset-turn approval.

Root independently recomputed recipe30inventory withdefaultinputs/nooverrides:9requiredresolved,0includedblockers,0unreviewedomissions.41historicalartifactpaths accounted:40replacementinputs and1explicitapprovedsource decision. Verified allsevenreviewedlanes289turn byte hashes/counts, fourstatic6138cases/32652approveddecisions, generatedsnapshotgzip bytes/decompressedcontent3887positive rows/3006programs,1630failurecandidates allineligibleDPO, nocheckedheld-sourcepositive. Snapshotasof14:37:23UTC; rawgenerationcontinues. RecipeSHA `3060b500525142bd1a8d36e3e1f4a49c801c956f50e3a4f34a9d7564bc409faa`, auditSHA `ed89bbc1ed1c16e419ce9e02aadf5832a01f86f0b77424cd3f0b9ec7518d79e1`; rootreceipt `runs/data-lineage-20260930/recipe-v30-review/root-review.json`. Rootindependentinventory `9f2e47a4-e861-4699-9ee5-79bf5c4bd66f.json` SHA `32520f0230f1ff5fedb21ec50b68b86339fae171631bb3bb27d77d99b028f80f`.

Recipe30 remains draft: studentrenderer/token/split/dedup/final25%reducermixpending. Unchanged350Mbuilderdefault isnotanapprovedstudentchoiceorMiniCPMreplacement. No tests/training/GPUevaluation. Rejectedstaticdraft3largefileslosslesslyarchived at14:37:32withoriginal/archive/roundtrip hashes: `workflow-quality-v4-stage/rejected-draft-archive-receipt.json`,2334968733bytes saved; all canonicaldata retained. Approved hardlinkedquality4largefiles/lineagereadonly0444 toprevent accidentalrewrites throughstagedaliases.

Monitor584649 has0alerts; teachers392661/483514/483515andwaitingcontrollers580812/554491unchanged. Nextfull **15:08:03UTC /17:08 Berlin**; rootactiveforinvestigation.

---

## Workflow typed reducers published — 2026-10-01 14:37 UTC

Published workflow **quality-v4** at14:34:56UTC after root exact partition/source/gold review and current-host native bundle validation. Default manifest SHA `bc6ee512e014177c2fb5b1b141bdfab3dcae12f1ad1fd9bf5fbcaf71f840b1f1`. **164exact source-bound directory-v1→directory-v4 replacements,0additions;4639othercases and27822otherturn byte rows preserved**. Replaced1136turns with1153new frozen22nativeapprovedturns. Workflow4803cases/28975decisions; allfourstatic6138cases/32652approveddecisions. SourceIDs/groups/revisions/license/split/source snapshots/oracle/golds unchanged; equivalent values checked through explicit job-key mapping. Oldquality3 and staged drafts remain auditable. Approved immutable large artifacts hardlinked into canonical data directory to avoid another3GBcopy. Do not mutate either linked path. Catalog new `workflowevals-quality-v4-publication` and default pointer ensure automatic carry-forward. Root receipt `runs/bonsai-followup-20261001-v5/workflow-quality-v4-byte-preserved-stage-v2/root-activation-receipt.json`; root partition approvalSHA `b52108dc685158474768a9c07caae0d669963d101a5029f1d314b9aa44398f54`.

Correction to prior14:21note: native turn source_ref is a canonical parsed-object digest, **not a raw file-line hash**. Mere JSON reserialization did not break that linkage. The first draft was held for exact unchanged-byte partition evidence/minimal publication, not a demonstrated stale source_ref bug. Rejected first draft's three large files are being losslessly gzip-archived with original/archive/roundtrip hashes and an explicit excluded-draft receipt; canonical originals and approved training data are retained.

Root verified supplemental full history inventory `identity-inventory-integrity-v4.json` SHA `505e89e15b94bc6ee1b8be6dc1d7b8f1d3efc216c0c095a6f28b2dc0d279fdf3`:12185files/22.696GB/31206Bonsairows;0malformed/vanished/readerrors.135partialprogramIDs,7source-resolved/128unresolved;512futuretargets/originalvariantIDs/sourceIDs have0saved/queued/partial collisions.89known dependency directory links and43nonartifactfilelinks explicitly outside scope; artifactlinks/otherdirectorylinks failclosed. Runtime-specific IR digest recomputation remains outside utility; root independent native review supplied it for targets. Launch-pinned original inventory unchanged. Reusable helper now streams hashes with parsed bytes, propagates pipeline errors, checks file identity/timestamps and records explicit symlink scope. Root added immediate rejection handler while preserving later awaited failure; syntax-only, no tests.

Bonsai/Luna teachers remain392661/483514/483515; waiting controllers580812/554491; monitor584649 healthy/noalerts. **Nextfull15:08:03UTC**. Recipe29 preserved historical; quality agent preparing fresh default-input recipe30/current-policy gzip snapshot, no training/GPUeval. Student render/token/split/dedup/final25%reducer gate pending. Runtime20/21/22 unchanged.

---

## Hourly sweep and next Bonsai campaign approved — 2026-10-01 14:21 UTC

Full 14:08:03 UTC sweep completed; next **15:08:03 UTC /17:08 Berlin**. Root remains active to investigate hourly alerts. GPU100%, container4.011GiB/6GiB,64GiB free. Bonsai delta32exactresults:29admitted/3rejected/124approveddecisions; main saved-output reader690rows/0blocking errors. Independent agent set-difference across all active readers:118new rows/113accepted/5rejected. These scopes differ. Canonical journals since13:07 have92finishes/zero new non-success finishes. New3Bonsai rejects repeat two legacy nested-answer wrapper failures and one EOF-only Markdown mismatch; two Luna rejects are tax-rate polarity inconsistency and substituting a different PO line. Golds and old IR remain unchanged. Neutral versioned prompt derivatives are isolated proposals, not live repairs or DPO negatives.

**Bonsai successor controller580812** now waits for current256queue392661 to finish. Root independently approved512fresh cases:92conventional directory +164official typed workflow directory-v4 +256primitives,256reducers/50%,977sourceidentities;2948frozen22approveddecisions,0reject/unlinked. All164 typed IR independently rederived from pinned source cache. Root separately streamed run results and14retained gzip snapshots; full pinned provenance inventory includes plain snapshots and135partial IDs too. No target/source/old-partial collisions. Approval `runs/bonsai-followup-20261001-v5/root-independent-approval-v2.json`, SHA `fb5d396875cca5cd92b9906833754e0a5252821b006ee7df9316697d4efde805`.

Combined queue `bonsai-with-reviewed-repair.queue.jsonl` has129entries/513cases, SHA `272529cf388064d64cb495e36262a224e1f7c95479a19068b25cb5ea3a4fa316`. Exact original member48 storage-loss repair first; preserve saved49–51. Plan `root-approved-successor-plan-v1.json` SHA `1d1e62a3f8e2b7da410afd20720823a6885375a3f59537686f5e1e70e101432b`, only permitted prior failed parent canonical hash30614a9…d8d. One supervisor/four requests/4800sec/40turns/384requests/runtime22; all618runtime files, current admission modules, helpers, immutable source/proof artifacts pinned. Mutable default static pointers intentionally excluded from launch pins so future reviewed static publication can advance them; underlying immutable source data pinned. No evaluation_fixture in any target or exact repair. Receipt `root-successor-launch-receipt-v1.json`.

Sixth exact same-queue storage-pause recovery approval registered for this future Bonsai campaign: `storage-recovery/queue-6-approved-v1-runtime22.json`, SHA `5e14a204ae545ba2c03ccf330692409388bb3ff3a98bbda3ed7c2e4324485901`. Existing five unchanged. Monitor now **584649**, launch-v5.json, supersedes565131; checks exact argv and liveness even before launch record exists. Both Luna554491/Bonsai580812 controller checks pass with0alerts. Running teachers392661/483514/483515 remain frozen20. Stop monitors, controllers and watcher for intentional shutdown.

**Static workflow quality4 NOT activated:** root byte-partition audit caught the first staged bundle reserializing all4,639 supposedly preserved records, despite semantic equality. That could break source-row hash linkage in retained turns. Original stage preserved; quality agent preparing new stage with exact original byte rows and exact candidate result bytes. Intended164 replacements are source/group/revision/license/split/gold-equivalent one-for-one, not164additive cases. Final student gates still pending; no tests, training or GPU eval run.

Inventory helper review caught parse-then-reopen hash race; new helper hashes parsed stream bytes and checks identity/timestamps. Error propagation and catalog-shape checks being strengthened; new full scan receipt pending. Existing launch-pinned inventory unchanged and independently reviewed target freshness remains valid. Root data audit initially exceeded its2GiB heap by storing entire891MBcorpus; corrected to retain only targeted512static rows/164factory records, then passed without any model calls.

---

## Source holds published; durable successor alerts — 2026-10-01 13:54 UTC

Hourly monitor is now **565131**, superseding490927 with reviewed successor-controller health alerts. Its minute heartbeat checks waiting-controller liveness/argv, exact plan binding, small policy/helper artifact hashes, and latest parent failed finishes against the exact approved failure hashes. Successful completed handoff permits controller exit; partial handoff does not. Member aliases and superseded older failures are excluded. Storage recovery is unchanged. Registry currently tracks Luna controller **554491**, planv6 SHA `e82f6635f79dbd08c6a23c34b731f0ba50aac3b853592fb398d9c206e0031c29`. Actual initial checks and latest heartbeat have no alerts. Receipt `runs/generation-check-20261001-resume/hourly-monitor/launch-v4.json`. Next full check **14:07:46 UTC /16:07 Berlin**; root stays active to respond.

Published exactly nine MuSiQue holds at13:38 UTC. Original source/gold/history and v24 artifacts preserved. Static directory **quality-v25:1212cases/3195approved decisions/972held decisions**; pointer SHA `aa1731ccf605856a9193119f706c765d5e095ea10650ed4b9e84746289f53f3c`. Source partition approval and activation receipt are under `hourly-health-next/bonsai-rejection-audit-1823/source-hold-publication-v4/`. Root independently derived exclusions from source IDs and checked exact retained byte sequences, all line hashes and 1221→1212 /4185→4167 conservation. Nine sources removed,18linked turns removed; no DPO negatives. Lisa role is supported; its hold concerns only the missing performer/video bridge. Tennis and Goshen remain unheld.

Prepared immutable **runtime22**, clone21 with ONLY source-review.ts/.js updated to eleven hold records (two previous plus nine newly confirmed); all618parentfiles verified. ManifestSHA `f1f42e622d1a79fd62463796153f5ced5b1a25248cf3a472d4713fbcf246c5e2`. Current workers **Bonsai392661 /Luna483514 /483515 remain20**. Stopped only waiting Luna controller500258 after confirming its successor launch record workers[] empty, then independently admitted/materialized all541pending invoice cases under current policy/frozen22:5862approved decisions,0reject/unlinked. New controller554491 rolls each slot at its own boundary; four exact failed finishes still excluded. Future541 storage approvals now22; current three queues keep20 approvals. Do not mutate frozen20/21/22 or their old plans/records.

**Recipe29 default-path draft** independently reviewed: four static corpora6138cases/32635approved decisions,3768selected generated trajectories/2910programs,1626failure candidates, seven reviewed lanes289turns. RecipeSHA `2aefa426b8bc207dc201d6f9cc8a68fe87177fa006c5d9b9cc31f1844fc54524`; root receipt `runs/data-lineage-20260930/recipe-v29-review/root-review.json`. Fresh current-policy snapshot manifest `1d0208fb-93b6-4531-ba73-fe218fe1a509.manifest.json`. Root streamed positives/failures and verified zero held-source positives and all failures ineligible as DPO negatives. All nine required inputs resolve with no overrides or included inventory blockers. Student render/token/split/dedup and final25%reducer mix remain pending; training has not started. Recipe default model is still LiquidAI/LFM2.5-350M; this is a draft, not a decision to train it or replace the candidate MiniCPM student.

Live compiled source-review was normalized after publication by an external host build: hash03843a44… versus frozen22a4fbf212…; root confirmed identical source-review records and all module bytes outside the nine inserted literal lines. Current source-conversion adds the previously reviewed strict derivative-identity guard versus frozen22; current collector adds an unrelated flat-program fixture branch. These differences are explicit in recipe29 audit and `runtime22/live-policy-equivalence-review.json`. Snapshot uses current admission policy; it does not certify byte-identical frozen22 execution. No frozen mutation and no tests/models/training in these audits.

Full Bonsai history inventory found the old folder-name filter missed results under `four-root.jobs/`; reusable provenance-based streamed scanner now covers runs and retained generated snapshots plus partial attempt IDs. After full exclusions only92fresh conventional directory cases remain (87CommitPack,2MuSiQue,2TreeDST,1TATQA). Next512 proposal is being revised to **92conventional reducers +164typed workflow reducers +256workflow primitives**, still256reducers/50%, via the existing official directory-v4 adapter. Legacy broad Record<string,unknown> workflow reducer contracts must not be generated afresh. Modern typed static conversion/lineage and exact member48 ENOSPC repair need root approval before new Bonsai controller launch. V3/v4 proposals are NOT approved. Current256Bonsai queue continues, so no immediate idle slot.

---

## Default carry-forward verified — 2026-10-01 13:12 UTC

Fixed `scripts/inventory_training_data.py` required-input accounting to recognize only catalog-approved replacements validated by the strict reviewed-input resolver, with hashes checked and the replacement actually included in the recipe. **Recipe28 uses default inputs, no explicit overrides**. Root recomputed the inventory and checked all seven reviewed lane byte/count hashes: nine required inputs resolve; 289 reviewed turns carried forward. Recipe SHA `55f7adc0dd131f35bf81f85f27f0752c03b36ab19a5dcef9af16c37b86f3164f`; root receipt `runs/data-lineage-20260930/recipe-v28-review/root-review.json`. Recipes26/27 remain preserved. Recipe28 is still a draft, pending nine source-hold publications and final student readiness gates. No tests or training run.

---

## Hourly sweep and follow-through — 2026-10-01 13:07 UTC

Root is actively following hourly alerts again. Monitor **490927** completed the 13:07:46 UTC sweep; next full check **14:07:46 UTC (16:07 Berlin)**. Bonsai **392661**, Luna **483514 / 483515**, and rolling Luna controller **500258** remain active. GPU was 98% utilized; approximately 68 GiB free. The hourly Bonsai supplement contains 28 new exact results, 26 admitted trajectories, 108 approved decisions, two rejects, and no missing results/source errors. Main reader scope: 572 saved aggregate rows and zero blocking reader errors. Empty active case210/673 outputs are pending, not failed finishes. Current Luna journals contain 209 and159 completed entries and only the four previously reviewed failed finishes. Use dated reports rather than stale flat authority counters.

Root independently approved the evolved v4 improvement partition: **66 retained / 10 excluded** of 76 physical turns, after checking all seven producer artifacts and declared computations. Approved manifest SHA `1fe3fbfbb32581b99ca1d4b78f52266ac0a45b854261bc05c10ec40cf45d9158`; receipt `runs/data-lineage-20260930/recipe-v26-review/root-v4-approval.json`. Alongside earlier source partitions, 43 physical turns are explicitly excluded and 289 reviewed turns retained across seven lanes. Originals remain untouched; no fabricated DPO labels.

Recipe27 is a draft with these 289 turns, four static corpora (6147 cases /32644 decisions), and the 12:34 model snapshot (3646 selected trajectories, 2806 programs, 1623 failure candidates). Its inventory uses explicit turn overrides; this does **not** prove automatic default carry-forward. Quality agent is investigating default required-input normalization and preparing a default-input proof. Nine confirmed MuSiQue source holds still await publication; student rendering/token/split/dedup/reducer-mix gates remain pending. No training started.

Proposed next Bonsai512 queue **v3 is NOT approved**: the independent root scan found previously attempted source `3hop1__478606_751065_78953`. Admission agent is correcting exclusion coverage in a preserved v4 proposal. Exact missing-member48 ENOSPC repair remains separately reviewed; preserve saved members49–51. Generic successor controller now follows authority PID across storage recovery and accepts only exact reviewed failed-finish hashes. Handoff claims, worker launch, and authority update share one lock, with a final predecessor identity/liveness check. Python syntax parsed; no tests run. No new Bonsai controller has been launched yet.

---

## Next-runtime guidance and evolving improvement lane — 2026-10-01 12:48 UTC

Frozen **runtime-v21** prepared by cloning20andchangingonly `dist/native/prompt.js` toaddthreecommittedAPPROACHbullets (relevantfieldselection, actualslicedelegation, minimalleafedit). Every618parentfilehashverified, onlyone changed; no compiler/oracle/recursion/tool change, JSsyntaxchecked, no tests/modelsrun. ManifestSHA **923b53a908903f0b2131cad29c75dc638955e52de44804ae005199a3b58f93e7**, receipt `runtime-v21/root-review.json`. RunningBonsai392661/Luna483514/483515remain20; don'tmutateeitherfrozenruntime.

NextLuna541nowreviewedfor21: all541currentadmitted/frozen21materialized,5862decisions,0reject/unlinked; receipt `luna-next-campaign-audit-v5/root-runtime21-reference-review.json`. Rollingcontroller **500258** replaces489449withoutanysuccessorlaunch. Planv5SHA **c28e883114701716abc2947e16876ece68e7f72850e883c6aa62861c7bd02bcd**; exactqueue/proofs/failurefinishhashespreserved. Future541storage-recoveryapprovalsruntime21updatedimmutably; current3queueskeep20approvals. NextBonsai512+oneinfrarepairproposalbeingpreparedfor21, modernMarkdowncontractsrequired.

**Newqualityblocker:** currentfolder-revisions-v4nowhas76rows/sevenproducers andincludesmigratedbadaggregation10rows, digest4c2eb950…dbf26c2 (sourcevalue.length, declaredsumcasesfail). Earliernotevalidv4empty-reductionhistorymustnotbeinterpretedasapprovaloftheevolvedentirev4corpus. Rootheldcurrentv4laneincatalog, pinningartifact; qualityagentprepared66-rowcandidate `folder-revisions-v4-reviewed-v1/verified-turns.jsonl` withunapprovedmanifest. Preserveoriginal76/candidates; fullsevenproducerreview+rootapprovalneeded. Recipe26draftcannotbereadyuntilthispartitionandninependingMuSiQueholdcandidatesareadjudicated/published. Newgzipmodelsnapshot3646trajectories/2806programs,1623failurecandidates; snapshotsselectionisnotfinalstudentreadiness.

---

## Sweep findings — 2026-10-01 12:35 UTC

Bonsai missingmember48 isconfirmedstoragefailure: `batch-0048.collector.log`SHA0383ccf8b2fc678da7fbfb161212b17bf299ea1fb567e8774c42896bbc1380be hasENOSPCwriteAtomic/worker/Promise.all. Batchfinish11:03:22 preserves49/50/51exactresults; only48mayberepaired, aseparateexplicitinfraexceptionin nextqueue. Notproviderstall orsemanticnegative.

RootindependentMuSiQueproposalreview narrows11to9sourcebridgeholdcandidates. Tennis2hop__85865_86706proposalmissedarticle2whichsaysDjokovicbeatFedererinallfourGrandSlams; goldsupported, modelNadalwrong. Goshen3hop1__374907_183029_831331staysreviewonly: multiplenameanchorsdon'taloneprovealternativevalidfullchains. Rootreceipt `source-hold-adjudication-proposal-v4/root-source-review-v1.json`; centralpolicy/staticsnotyetchanged. SourceholdsarenotDPOlabels.

Futureimproververificationguardbugfixedminimallyin **untracked unrelated** `ts-host/scripts/self-improvement/verify-improvement.mjs`: declaredtrain+validationquality1/gatesnowrequired, instead ofmerelyrecordingfailedtrainstatusandcomparingvalidationtohistoricalscore. Syntax-only `node --check`; no modelreplays/testsrun. Roottwoeditsleftuncommittedtoavoidcommittingentireunrelatedfile. Receipt `folder-turn-source-review/verifier-admission-fix.json`pinspatch/beforeafterhash. Existingqualityagentrefreshingcurrentpolicygzip/modelsnapshotanddraftrecipe26, preservingalllanesandindependentlyauditing33held/59retained; source-holdpublication/readinessstillpending.

---

## Durable storage recovery enabled — 2026-10-01 12:22 UTC

Monitor **490927** supersedes389123 (`hourly-monitor/launch-v3.json`). Root approved future storage-pause recovery for **five specific queues**: bothactiveLunav4queues, currentBonsai256queue, and bothreviewedLunav5successors. Approvals/pinnedhashes are under `runs/generation-check-20261001-resume/storage-recovery/queue-*-approved-v2.json`, configured in authority `storage_pause_recovery`. Recovery is limited to latestrecordedstoragepause onanunfinisheduniquequeuekey, absentPID/exactcommand, unchangedqueue/supervisor/all618frozenruntimefiles/exactauthoritybinding; requiresatleast4GiBfree. Capturespauseevent/journalhash/oldnewPIDs beforeclaim; crashafterclaimrequiresreview. Stops/otherfailures neverauto-resume. Disablethisconfiguration/stopmonitor forintentionalshutdown.

Rollingcontroller **489449** supersedes484210 afterreviewedpendingclaimwithnoworkers. V4planSHA **8437a61ac54ed240d71d0b42f30a460004fd2b93b9bd8fb1769de8cbaa0c0bd6**. FollowscurrentPID byexactpredecessorqueue/journalbinding, soapprovedstorageresume doesn'tbreakhandoff; rechecksworkerabsenceunderlock. Fourexactreviewedfailedfinishhashesstayexcluded. Newunreviewedincompletefinishesstillblockforrootreview. ActiveLuna483514/483515; Bonsai392661; watcher368408. Nextfull13:06:58UTC.

Root approved legacyreview partitions:59v2turnsretained/10aggregationheld; all23oldfolderrevisionsturnsheld. NewrootapprovedmanifestsSHA8fca567d48307cd3162b6de78ffbcb25fd27ec45d84a85a5c6694e11aafded55 andf305beca5b9dc7150c7b4340b44f2d7c1b4cca88e085502fb7238fdb92b2ead8. Exactphysicalrowpartitionsandallproducerhashes/fullcaseinputcomputationsverified; originalartifactsuntouched, noDPOlabels. Fixedresolverapproved-statuslookupbugandcatalogpins. Current-manifesthistoricalpointerupdated; recipe25remainshistoricallyblockedandfreshrecipe/readinesspending. Onlyownedguard/builder/cataloghunkscommitted, unrelatedAPIchangeedgesremainworkingtree.

Next-runtimepromptguidance committed: selectrelevantfieldsfromlargeinputs; passactualdisjointitem/pageslicestohelpersinsteadofsamefullpacket+PAGElabel; deepcopytree/editrequestedleaf/preservewrappersandsiblings. Frozen20unchanged, soongoingjobsdon'tyetreceivethesefutureguidancechanges. EOF-onlyMarkdownreject74confirmedbenignexistingboundedvariant; oldqueuedIRlackednewcontract. NextBonsaiprep mustapplyexisting `applyScopedMarkdownEditContract` beforequeuecreation, preservingoriginalIR/sourceproofandnewcanonicaldigest. Do notrewritehistoricalprompt/historytoartificiallypromoteoldrow. Next512freshBonsaicampaignbeingpreparedwithreducerspriority; rootapprovalrequiredbeforelaunch.

---

## Recovery and hourly follow-through — 2026-10-01 12:07 UTC

Mechanical audits continued at09:43/10:43/11:43 but root ended its active turn and did not act on hourly-agent escalations. Luna workers376000/376001 stopped at storage floor around11:00; filesystem briefly exhausted, later recovered to71GiB. Rollingcontroller389122 failedclosed on incomplete predecessor rather than launch successors. Do not describe those queues as completed.

Root restarted **483514 /483515** on the SAME original v4 queues/journals at12:04 after verifying unchanged pinned artifacts and storage-pause evidence. Saved completed and failed attempts are skipped; unattempted work resumes. Evidence `runs/generation-check-20261001-resume/luna-storage-resume-1206.json`. Bonsai **392661** is active on256-case successor, GPU98%. Runtime-v20 remains frozen.

Rolling controller **484210**, plan `luna-next-campaign-audit-v5/root-approved-successor-plan-v3.json`, SHA fbd4c64cfa876a6501dd4dd6606130f6a81c5a9baa59993236f478fb51187f82, binds resumed PIDs and allows exactly four reviewed empty failed finish records by canonical event hashes. Failures remain excluded and preserved. All other incomplete outcomes still require review; no false positive accounting or blanket failure bypass. Old v2 claim/failedcontroller evidence retained. Root is preparing approved automatic same-queue storage resume in monitor; not yet enabled.

Requested immediate sweep completed12:06:58UTC: zero blocking reader errors/futureholds; Bonsai delta8exactrows/8admitted/65decisions/zero missing. Main scope438savedrows; don't confuse this with new cases. Reports `hourly-health-next/health-2026-10-01T120658135Z.json` and matching spool; nextfull13:06:58UTC. Existing timeout/admission agents renewed for active failure/UXreview; root must remain active and act on alerts instead of ending after setup. Missing Bonsai member48 from earlier partial batch and oversized invoice paging loops are under investigation. Stop monitors/controllers/agents plus workers for intentional shutdown.

---

## Continuous utilization — 2026-10-01 08:43 UTC

Bonsai recovery supervisor **332364** remains active with four concurrent requests; GPU sampled at **98% / 7,457 MiB**. At 08:43, 29 of 37 recovery parent batches had finished; the current batch was producing fresh replies (29 saved replies, zero repeated request hashes). These are batch counts, not case counts. Two Luna workers **376000 / 376001** remain active on the 1,024-case campaign; journals showed 52 and 27 completed queue entries respectively, with fresh replies on both active cases.

Reviewed work is staged for automatic continuation: Bonsai **256 cases** (192 reducers, 64 workflow) via controller **389121**, plan `runs/bonsai-next-20261001/root-approved-successor-plan-v2.json`; Luna **541 fresh invoice cases**, split 271/270, via rolling controller **389122**, plan `runs/generation-check-20260930-hourly/luna-next-campaign-audit-v5/root-approved-successor-plan-v2.json`. Each Luna slot rolls forward after its own predecessor completes; it does not wait for the other worker. Controllers verify pinned artifacts, policy, runtime and complete predecessor accounting before launch. Stop these controllers during intentional shutdown. Do not restart their claimed queues blindly.

08:43 hourly audit completed with zero blocking reader errors/future held queue rows. Bonsai delta:48 exact saved cases,46 admitted/2 rejected,134 approved decisions,zero missing/source errors. Reports under `hourly-health-next/health-2026-10-01T084316332Z.json` and matching canonical spool. Next full audit October1 09:43:16UTC; use authority next_check_at for scheduling. Disk unexpectedly fell to1.3GiB; root removed1.451GB inactive Homebrew/pnpm/thumbnail caches older24h after no-open-handle check, restoring approximately2.8GB available. Full per-file audit `runs/generation-check-20261001-resume/cache-space-recovery-v2.json`; originals/models/training untouched. Hourly agent investigating underlying disk growth.

Mechanical monitor **389123** checks availability every minute and runs full audits hourly. Missing-server watcher **368408** remains active. Existing Luna timeout-audit agent is assigned ongoing hourly generation/rejection oversight. Shared `scripts/generation_authority.py` lock now coordinates monitor and successor authority updates, preventing concurrent writes from losing newly launched worker IDs. Python syntax parsed; no tests run. Policy changes must explicitly refresh successor approvals rather than silently breaking pinned plans.

Legacy quality review found **all 23 old folder-revision turns unsafe**, plus 10 historical folder-api-v2 aggregation turns. The old empty-sum producer retained code without an initial reduction value; its validation subset omitted the failing empty-list input. Fresh v4 code actually fixed it and is separate history. Preserve originals and keep candidate replacements unapproved until root verifies full source/case evidence, row partitions and catalog graph. In particular, historical replacement chains must not redirect held/retained old partitions into unrelated fresh v4 histories. No DPO negatives inferred from these flags.

---

## Current utilization — 2026-10-01 08:12 UTC

Both teachers active. Luna **376000/376001** launched at08:05:24UTC on **1024freshworkflowcases**,512each:979invoice/34security/11observability. Root independently checked everyIR/gate/static reference,1024currentnative admissions/10857static approved decisions,0unlinked, and6538savedartifacts/7162Lunarows withzero target matches by both source identity fields. Final activation approval `luna-next-campaign-audit-v4/root-launch-approval-v3.json`, launch `launch.json`; onlyuseworker-{1,2}-v2queues/journals. V1emptydiagnostic preserved: initialprep hitNodeInvalidstringlength on851MBreadFile; thenmisusedadmitRow onincompletewrapper, falselyselected0. V2streamed/chunkedprepvalid; reportv3fixesv2'sstaleproofledgerpointer. Prior256batch250accepted6rejects; fullinputauditclassifiesall6asmodelerrors,no goldchange (`luna-next-campaign-audit-v3/reject-audit-v1.json`). NextLunastandbyproposalpreparationexcludesall1024queuedidentities andallhistory.

Bonsai332364 continues145caserecovery,fourrequests,GPU98%. **Reviewed automatic successor staged**:256cases,192reducers(180CommitPack+12TreeDST),64workflow,64fourcasebatches. Root nativeaudit256/965staticdecisions,0reject/unlinked andindependent3560artifact/5721Bonsairow source/IR scan; all145future recoverytargets excluded. Approval `runs/bonsai-next-20261001/root-approval.json`; manifest-v2. Originalmanifestmistookraw JSONLlineSHA forcanonicalIRdigest; v2distinguishesbothandpreservesalloriginalbytes/queue. Officialstreamedutility `node ts-host/scripts/record-digests.mjs --input PATH --runtime HOST_ROOT [--field task.program_ir]`; usecollectorrecordDigest, neverapproximatecanonicaldigestwithrawlinehash.

Successor scheduler **378020** waitsforrecovery332364 thenlaunchesonlyafterall37parentbatchesfinishwithcompleteexactoutputaccounting. Plan `runs/bonsai-next-20261001/root-approved-successor-plan.json`, SHA d21aefe456fe6f43d438af3c35a7ced633f790370aa530e1d38b2034c34650a0. `scripts/start_reviewed_generation_successor.py` rehashesqueues/IR/rootapproval/currentpolicy/supervisor/all618frozenruntimefiles andclaimsoncebeforelaunch; partial/incompletepredecessor/policydriftblocksforagentreview. Launchrecord `automatic-launch.json`; newqueueauthority/journalupdatedafteractivation. Do notlaunchsamequeueagain. Stopthisschedulerwithothermonitorsduringintentionalshutdown.

Persistenthealthmonitor367674 andmissing-serverwatcher368408 active; fullcheck08:42:36UTC. Diskrecovered1.416GiBbyremoving13unused downloadableElectronzipcaches(age>24h,nopenhandles), no training/model/lineageartifactremoval; audit `runs/generation-check-20261001-resume/cache-space-recovery.json`. Approximately5.5GiBfreeafternewsnapshot.

Qualityissueunderrepair: recoveredmissingfolder-api15turnartifactisrealproduceroutput; newdefaultfolder-revisions23turnlanecarries12unsafezero-fallbackturns(oldproduceraccepted=true despitefailedrewriteandwrongfinalcode). Source-oracleagentpreparingexplicitproducer-boundfilter/replacement/defaultpipelineguard. Recipe25/currentfreshsnapshot exists butdoNOTclaimtrainingreadinessuntilguardandrootreview. Preserveoldbadproducerhistory; nodirectDPOlabels. Freshsnapshotcurrentpolicy3198positives/1604failurecandidates; source82mustbeledger-heldandabsentpositives. Source-awarefullrecipeauditpending.

---

## Current checkpoint — 2026-10-01 07:42–07:49 UTC

Bonsai332364 healthy, GPU98%,17completed recovery batches/68exactresults:55currentadmitted,13rejected,277approved decisions,0missing/sourceidentityerrors. This is cumulative recovery; since05:28,64results/52admitted/12reject/256decisions. Remaining77recoverycases; no transport/supervisor failures. Full health `hourly-health-next/health-2026-10-01T074236085Z.json` and matching canonical spool supplement;0metadata readererrors/futureholds. Next full check08:42:36UTC.

Both Luna workers333669/333670 finished all256workflow cases normally:250rawaccepted/6rejects. Updated authority/state marks queues completed. Existingquality agent prioritizes next256fresh workflow proposals before fullinputrejectionaudit. Root independent approval required before launch. Existingadmissionagent prepares successor Bonsai campaign, excludingall prior attempts and all145currentqueuedrecoverytargets; do not leave teachers idle once approved eligible work is ready. Neither idle Luna nor fully consumed queue implies all corpus work complete.

Persistent hourly health monitor **367674** runs `scripts/monitor_generation.py`, launch/status/journal under `runs/generation-check-20261001-resume/hourly-monitor/`. It checks worker availability every60sec and fullhealth/spools hourly; records `needs_agent_review` for idle workers. It does not independently perform AI repairs or launch unreviewed queues. Missing-server watcher launch recorded in `server-watch-launch.json` there: four slots/53248context, singleton lock, exponential restart delay; restores only absent container and never kills live/loading container on health failure. Stop both monitors explicitly during requested shutdown.

Quality24 catalog/root hashes and allfive v23→v24 edges independently verified; everyprior replacement edge retained. Recipe24 is a draft carrying allfourstatics/twolegacy/model/failure lanes, but STALE model snapshot still includes held source82 (positive line488) and does not cover currentgeneration. Missing historic folder-api-v1 verified-turns artifact also blocks inventory; no ready record/training. Existing source-oracle agent refreshing current-policy gzip snapshot/recipe and investigating recoverable missing lineage. Disk4.7GiBfree; no deletions/plain large snapshots. Continue preserving unrelated API/program-improvement work.

---

## Current checkpoint — 2026-10-01 05:28 UTC

This section supersedes historical live-process and publication notes below. Bonsai server was absent on inspection; previous server log shows orderly cleanup, but the cause is not established. Original 314-case campaign preserved 169 exact saved results (103 raw accepted, 66 raw rejected); 37 unfinished batches exhausted through transport failures after server loss. These are not 37 semantic failures. Recovery contains only the 145 unsaved cases, in 37 batches, with no reruns of saved positives or negatives.

Server restarted with four slots, original 53,248 context/6 GiB container settings. Recovery supervisor **332364** runs frozen runtime-v20 with four model requests, 40 turns/384 requests per batch and 4,800 seconds. Root approval, pinned queue/source hashes and launch metadata: `runs/generation-check-20261001-resume/`. Journal: `bonsai/journal.jsonl` there. At 05:28 GPU utilization was 99% and fresh saved model replies were increasing; first recovery batch still active. Supervisor now waits for local server health before consuming an attempt and applies exponential failure backoff to local Bonsai as well as Luna. Python syntax parsed; no tests run. Frozen runtime unchanged.

Two Luna workers **333669/333670** launched at 05:27:41 UTC on **256 fresh workflow sources**, 128 each: invoice 78, customer service 50, security 64, observability 64. Use only `luna-next-campaign-audit-v3/worker-{1,2}-v2.queue.jsonl`, journals under corresponding `worker-{1,2}-v2/`; approved launch `launch-oct1.json`, independent review `root-approval-oct1.json`. Root scanned 6,000 saved files/6,650 Luna rows using both source_ids and external_source.source_id, found zero prior selected-source attempts, and verified all IR/gates/queue/runtime hashes. These add model strategies to already static workflow cases. Earlier drafts remain unlaunched. Previous 106-case batch completed with 99 current admissions/480 approved decisions and seven rejections.

**Static quality-v24 published** with 1,221 cases/3,204 approved decisions, all four static lanes now 6,147 cases/32,644 decisions. It preserves 1,220 prior IR records unchanged, removes unsupported MuSiQue source82 and ambiguous signed-P&L original374c, and adds the explicit exact signed-P&L derivative. Manifest SHA256 `12ec245d4c203c0ef109ec8e81849c563b38eb584ac8794fc8c9fbd40e2bfa05`; root approval under `quality-v24-review-candidate-v3/root-approval.json`. Original artifacts/golds/history retained, neither ambiguous source is a DPO negative. Catalog carryforward and recipe-v24 refresh delegated to the existing source-oracle audit agent; recipe23/inventory-ready remain stale until refresh. No student training or final renderer/token/split/dedup/reducer gate claim.

Rejection audit continues: existing admission/DPO agent reviews original Bonsai rejects indices124–168; earlier narrow-alias/source-hold proposals remain root-review-only. Do not blanket promote those proposals or blindly retry failed sources. Check current authority `runs/generation-check-20260930-hourly/check.json` and `runs/luna-file-tree-20260930/state.json`; spool reader now follows authority bonsai_journal. Preserve unrelated program-improvement/API migration edits and do not commit them with generation changes. Disk approximately 5.7 GiB free before launches; use gzip-only recipe snapshots.

---

## Current checkpoint — 2026-09-30 19:58–20:04 UTC

Bonsai223773 continuesfourrequests/GPU100%;124/314saved at19:57,72rawaccepted52reject,31finishedbatches,0non-success. Canonical delta since14:09:100exactresults,56currentadmitted,396approveddecisions,0missing/sourceerrors. Health `hourly-health-next/health-2026-09-30T195844728Z.json`; canonicalspool `bonsai-canonical-job-spool-supplement-asof-2026-09-30T195844728Z.json`. Nextfullcheck20:58:44.728UTC.

Luna261656/261657completed53cases each:99/106currentadmitted,480approveddecisions,0unlinked;7rejects(6SciFact,1observability). Audit `luna-next-campaign-audit-v2/root-completion-audit-1957.json`. No liveLunaatthischeckpoint;next256freshworkflowcasequeuesbeingprepared, including bothhistorical source_ids and external_source.source_id and excludingALLpriorattempts. Correction: prior106launchhad94trulyfresh+12previouslycoveredSciFact sources because root/agent scans missedexternal_source.source_id; originalzero-target-match proof had that limitation. Preservecorrections/ledgers v7/v8 and neverclaim106fresh.

UrgentcentralholdcompiledforMuSiQue3hop1__642758_643936_45121 (source82): birthplaceMiddelburghomonymdoesnotestablishSouthAfricancountrybridge; a historicgold-matchingpositiveexists andmuststayexcluded. Rootreadsourcearticles16/17/15, addedhold, Nodebuildpassed/nativeadmissionholdcheckpassed. Recipe23/inventoryreadyareSTALE;quality-v24candidate revisionremovingthissourceandrecipecarryforwardpending. No trainingstarted. Rejectionaudit1823classifies35rejects:manysupportedfullphrases,9source-reviewproposals,3potentialgenuinemodelerrors; latest102–123auditunderway. Do notuseunsupportedchains/blockedrowsasDPOnegatives orblindlyretryfailedsources. NextLunabatchisworkflowonlywhileMuSiquesourceproposalsareadjudicated.

## Current Luna campaign — 2026-09-30 14:32:54.125977 UTC

Two Luna workers **261656/261657** now run **106 fresh source cases** under unchanged frozen-v20,53each:50SciFact plus14each of invoice processing, customer service, security incidents and trace observability. Workflow cases already have static training references; new model runs add strategies. Launch/root approval under `luna-next-campaign-audit-v2/`; root independently scanned6438saved Luna rows across5788files and found zero selected-source attempts, verified every selected IR/gate/queue hash, and excluded successful same-source coverage even when IR differs. Do not launch the superseded128case proposal. Existing exponential backoff/request limits retained; Bonsai223773 continuesfourrequests.

Source-aware inventory of the6025published directory/workflow cases:3534raw-success same-source Luna coverage,1952freshsources,233failed-onlysources deferred,303gatedholds,3source-chain reviews. These are model-coverage units, not final admitted training counts or a requirement to rerun already static data. Exact-IR matching alone falsely made219TATQA cases appear untouched. Initial v2 ledger also omitted late-added dispositions; root's independent fresh-target proof permits this106case launch, and a corrected immutable full ledger/report is being prepared. Never equate the old report field successful_exact_luna_matches with its same-source count.

New source-conversion guard rejects operation derivatives whose external source identity remains their parent or disagrees with source_ids. Node build and native draft/corrected identity audit passed; no tests run. Quality-v24 candidate replaces the ambiguous signed original with its explicit derivative, carries1221unchanged cases, and preserves lineage. Latest audit `quality-v24-review-candidate/quality-v24.current-policy-audit-v2.json`, SHA473c79e3c02a8d35362150e070fec7c7e03f5748d04de65ad328452f554266fd; publisher `publish-quality-v24-audit-v2.mjs` preflight passed but publication/catalog/recipe refresh remain pending. Prepared clarified5688derivative queue remains unlaunched under `tatqa-5688-row-clarification-review/`. Four finite MuSiQue answer aliases and three source-chain hold proposals are review-only under `hourly-health-next/bonsai-rejection-audit-1409/review-proposals.json`. Next full checkpoint15:09:19.749UTC.

## Luna resumed; backlog review underway — 2026-09-30 14:14 UTC

Follow-up14:15: both tiny queues completed. Signed derivative admitted3model decisions; unchanged maturity question quiesced (model distinguished payables from lease payments),0admitted. Preserve this failure, no further blind original retry; explicit maturity-row derivative is being prepared. Broad source-aware campaign preparation is underway. Central sign-convention hold374c4ad9 is now applied and compiled; quality-v24/recipe refresh pending. Initial Node build caught unrelated concurrently edited scoped-fs importing an as-yet-unwritten folder-iteration module; after that file appeared, the Node build succeeded. Those other source changes remain untouched and outside this task's commits; frozen-v20 is unchanged.

The overall generation target is not complete. Two fresh Luna supervisors **257261/257262** run signed-P&L and maturity-row repair cases under frozen-v20, one request each, existing exponential backoff, output-accounting/2. Launch/root-independent approval: `tatqa-signed-pnl-1158-review/fresh-luna-queue-v20/`. The signed derivative fixes synthetic external source identity (v2 had incorrectly retained its parent's source ID); evidence/gold/groups retained, exact -17.1. The maturity repair uses the original rejected Luna IR/prompt/inputs unchanged, expected600; reference success is not a model success or automatic DPO pair. Broader eligible Luna coverage is being audited; replenish only with reviewed missing coverage or justified repairs, not repeated successful cases.

Full checkpoint14:09:19.749UTC: Bonsai alive,97%GPU, zero reader errors/non-success finishes/future held roots. Exact completed-batch spool24results,16admitted,8rejected,111approved decisions,0missing/sourceerrors. Partial active aggregates can also include unfinished batches; do not conflate their counts. New rejection audit underway. Next full check15:09:19.749UTC. Health reader initially misclassified73not-yet-written outputs of unfinished queue entries as errors; preserved report14:08, fixed disposition, corrected report14:09. Missing outputs for finished entries still block. The spool reader now pins the current authority queue/runtime and the report's baseline instead of historical fixed paths.

## Live status verified — 2026-09-30 13:17 UTC

Bonsai PID223773 is alive with four concurrent requests. GPU sample98%,7457/8188MiB; journal records fresh replies and increasing decode totals. Previous queue completed normally, and the current314-case campaign began12:51UTC. Luna's three calculation cases are complete; no live Luna generation workers. Next full checkpoint remains13:25:51.941UTC.

Recipe-v23 carryforward/inventory audit passed16checks:2679selected trajectories/2129programs, allfourstatic bundles6148cases/32645approved decisions, bothlegacy reference sets, and1511retained failure candidates with no negative labels. Audit/readiness under `runs/data-lineage-20260930/recipe-v23-failure-inventory-audit/`. This is inventory readiness only. Pending source374c4ad9-300a-44b0-95bc-77f05de463b0 sign-convention review must be resolved before final readiness; no student training started. check.json now reconciles these counts and explicitly tracks the pending review.

CommaQA's one-case Bonsai queue remains prepared, unlaunched, for the next available campaign boundary. Keep the current immutable314-case queue and four-request concurrency. The TreeDST timeout candidate already has an accepted exact retry; do not duplicate it.

## Live generation restored — 2026-09-30T12:51:30.276969Z

Bonsai **223773** now runs verified frozen runtime-v20 at **four requests / four-case batches**, output-accounting/2, current source policy. **314 model-diversity targets**:212MuSiQue/102SciFact, excluding79cases with prior savedBonsai coverage and2unresolved sourcecases. Existing Luna trajectories are deliberately allowed; these are additional teacher trajectories, NOT314newunique cases. Ten exact reviewed wording variants prioritized first. Queue79entries, per-case40turns/384requests and batch4800s. Newlogs/outputpaths, priorjournal preserved; oldexhaustedqueue/PID110507notrestarted. Launch/rootqueueapproval under `bonsai-model-diversity-v20/`. GPU98% afterstart,7457/8188MiB.

The prepared agent plan’s “byte_identical_to_published IR” flag was too broad:10oracle-only IR variants differ intentionally; their source evidence/unaffectedfields remain identical. Rootqueueapproval corrects scope; publishedstaticbundle unchanged. Separate timeoutaudit incorrectly calledTreeDSTmissing bycheckingonlyoldspool; acceptedexactretryexists, so no rerun. CommaQA eligibleBonsai-diversity work remains separately prepared, not inthis314queue.

Recipe23published/planning selection2679rows/2129programs; fullcarryforward/failure/inventory audit underway, notfinaltraining readiness. Snapshot includes10approvedMu trajectories/37decisions,1definition/3 and3fresharithmetic/12 =14rows/52decisions. No live Luna generationworker afterthreecasecompletion. Nextcheck13:25:51.941UTC. Newrejectionaudit foundoneadditional sign-convention ambiguity (source374c4ad9...) andone genuine wrongmaturityrow (5688b0ff...); hold/clarified signedcase and samepromptsuccess lookup areprepared fornextqualitycheckpoint. NoautomaticDPOnegative or numerical target injection.

## Exact calculation pilot completed — 2026-09-30 12:46 UTC

Two Luna workers220765/220766 completed three explicit-operation cases under verified frozen runtime-v20 (618files, manifest956f105f157d8dc42b4941f780d154f4a1e3e4463275e939f561cc5a62d06a53). Current native admission/materialization accepts all3/12decisions/0unlinked. Actual outputs0.7million,4million,7.0million; exact decimal mode permits7.0=7 and rejects nearbychangedvalues. All three jobs have exact output-accounting/2 completion proofs. No live Luna worker; do not restart these completed queues. Artifacts/launch under `tatqa-operation-derivatives-v3/`; original ambiguous source IDs remain held.

Root independently approved the exact source-clause definition alternate as one unchanged historical Bonsai trajectory/3decisions, fullvisible-input/output/file/history/provenance comparison passed. Original collector file version1 retained; current comparison version2 recorded separately. Artifact `tatqa-definition-alternate/root-approved-definition/definition-source-alternate.result.json`, SHA9625a2625e9a9099fb7b9ac9200600dfec690ea1fc9bbe9bc0bdb5be51f61fde. No model answer rewrite. Four current static6148cases/32645approved decisions, including publisheddirectoryquality23. Recipe23refresh nowrunning; audit/readiness pending. Bonsai nexteligiblecross-teacher sourcecoverage queue being finalized; GPU remains idle untillaunch. Next hourlycheck13:25:51.941UTC.

# Current source-curriculum publication — 2026-09-30

Directory-expansion quality-v23 is published at `data/teacher/directory-expansion/static.manifest.json` (SHA-256 `7778ba25d1d5ca0828de23f589c9de74429f86d9a1d9e90c3fa7fa8f1a4db8c6`). It contains 1,222 cases: 287 commitpack, 219 TATQA, 241 MuSiQue, 300 QASPER, 154 SciFact, and 21 TreeDST; 4,187 materialized turns, 3,205 approved decisions and 982 held decisions. Three root-approved TATQA source holds remain explicit and are not DPO negatives. The original MuSiQue source hold remains; one exact reviewed Oklahoma annual-event variant is included.

`training/data_sources.json` now maps quality-v21 and quality-v22 artifacts transitively to v23 while preserving every prior replacement link (catalog SHA-256 `774cbb3941c210b28150a314d030a041544b59f6ed58c0a74aff0b9ae311787c`). Post-publication `staticBundleInput` and current-policy hash verification passed; receipt: `runs/generation-check-20260930-hourly/quality-v23-review-candidate/quality-v23.postpublication-verification.json` (SHA-256 `256a059dde4dcfbcb82ea0963844db2a4da7bcc17e423e376ab303edccb628c8`).

This publication is **not training-ready**. Root must refresh recipe23, then complete the recipe-bound corpus, template/render, split, dedup and reducer-ratio gates before selecting or starting student training. Frozen runtime-v20 is already pinned at `runs/generation-check-20260930-hourly/runtime-v20/frozen-runtime.json`; do not modify it. Previous handover notes below are historical and may describe earlier manifests, runtimes, or queue states.

---

## Hourly checkpoint — 2026-09-30 12:25:51.941 UTC

Bonsai PID110507 exhausted its current queue and exited normally around12:21:25UTC; GPU0% at checkpoint is explained by completion. No live generation workers until next qualified campaign launch. Do not restart exhausted queues. Preparing eligible coverage audit/current-runtime campaign; server35762 remains available. This hour96journal finishes =40Bonsai parent batches+56member aliases. Exact spool reconciliation:50distinct saved results, allID/digestmatched,48current admitted/materialized,2rejected,221approveddecisions,0missing/sourceerrors. Main reader0errors/0non-success finishes/0futureunstartedholds. Reader scope is explicit; journal aliases are not cases. Reports `hourly-health-next/health-2026-09-30T122551941Z.json` (SHA2072ce58f1ee2478a32b7a594eed2b1a28d3e9b1f0e39a73fa4889f7e1baaaac) and `bonsai-canonical-job-spool-supplement-asof-2026-09-30T122551941Z.json` (SHA0b014380…c03). Nextcheck13:25:51.941UTC.

Three TatQA operation ambiguity holds now current/compiled; staticquality23replacement is prepared, not published. Recipe22/inventory22 stale. The text-record return/file consistency gap is fixed in candidate code and undergoing native controls; selected222textrows already have matchingreturns/files. New explicit-computation derivatives must use exact decimal value comparison, not inherited2dp-equivalence; wrong nearbyvalues mustfail. Existing declared2dp display cases remain separate pending source/accepted-output audit. No numerical target injection.

## Builder contracts integrated — 2026-09-30 12:14:39 UTC

The future source builder now applies the ten source-pinned output equivalences and the exact Oklahoma three-event variant after source-group assembly. Isolated actual builder replays matched all ten registry records plus the Oklahoma full-record digest; an unrelated source remained unchanged. The Node host build succeeded. Oklahoma original remains held; only the complete pinned variant is exempt. Native reference/admission/materialization passed, with all three source-listed answers accepted and changed-record/unsupported-answer controls rejected. Frozen runtime19 remains unchanged.

Evidence: `musique-title-path-r2-next/root-review-needed/revision-4-oracle-review-candidates-v3/builder-integration.audit.json` and `musique-oklahoma-event-prototype/builder-native-audit.json` under the hourly run. Historical review-origin fields inside the immutable equivalence metadata remain preserved; this dated entry records current builder approval. Recipe22 remains stale by the ten already approved teacher trajectories; Oklahoma static carryforward is being prepared, not yet published.

Exact-input computational outputs must remain exact. Full-source calibration corrected all three apparent Bonsai calculation defects to operation ambiguity (relative versus amount change, annual-total versus four-cell mean, signed versus absolute difference). Preparing explicit-operation new cases and source holds for ambiguous originals; no answer injection or DPO negative labeling. The scale omission and faithful definition span are separate issues still under review. Next full checkpoint remains12:25:36.838UTC.

## Reviewed oracle-only carryforward — 2026-09-30T11:54:32.489Z

Root independently verified and published **10 unchanged teacher trajectories /37 admitted decisions** under `runs/generation-check-20260930-hourly/musique-title-path-r2-next/root-approved-oracle-equivalences/`. Finite, source-pinned aliases preserve original golds, actual model outputs, files, complete histories, provider requests and runtime19. Root compared the full original/candidate payloads and native oracle/admission/materializer results. Originals remain rejected under their historical narrow oracle, with both artifacts preserved. No answer injection, general fuzzy rule, numeric value tolerance or history-replay gate exception.

Canonical parsed-trace hashes match original provenance10/10. Raw trace JSONL bytes have separate hashes; comparing those two hash domains had created a false integrity alarm, now corrected in v3 audit. Futurebuilder exact-record alias integration + a single full-record Oklahoma3-event contract exception are being prepared/audited; do not mutate frozen19. Recipe22 predates these10publishedresults, so next hourly snapshot carries them. Nextcheck12:25:36.838UTC; Bonsaiongoing, no Luna generation worker until next qualified queue.

Five Bonsai rejects are under source-operation review. Earlier “two calculation errors” label is provisional: OtherCosts2% is a displayed relative-change metric versus0.7amount;8amendments averages annual totals versusgold4averagingfourcells. Actual arithmetic may be right for a different interpretation. Signeddifference±7 is also ambiguous; do not create DPO negatives without a clear operation contract. Scale omission and faithful text-span variations remain separate.

## Accounting clarification — 2026-09-30 11:47 UTC

The 11:25 checkpoint recorded **138 journal finish events**, including parent/member aliases. Exact spool reconciliation finds **60 distinct Bonsai case results**, 55 currently admitted/materialized (237 approved decisions), five rejected; all60exactID/digestjoined, no missing saved results. The other15distinct model results are13title variants and2derivatives. Thus **75 distinct case results**, not138newcases. Evidence: `hourly-health-next/bonsai-canonical-job-spool-supplement-asof-112536838Z.json` under the generation-check run. Empty working aggregate export did not mean missing saved jobs. Both partial/failed reader reports are preserved with a scope correction and future error/completeness gates.

Recipe22 inventory audit passes16/16carryforward/hash checks;2,618selectedteacherrows/2,107programs and1,509retained failure candidates, allnegative-ineligible. Fourstatic6150cases/32667approveddecisions andbothlegacy792/1400sets verified. Readyrecord `runs/data-lineage-20260930/data-inventory-v22.ready.json` validatesagainstcurrentpolicy; inventoryonly, notfinalstudent readiness. Additional exact freshderivative-selection proof supplement pending. Nextfullcheck12:25:36.838UTC; Bonsaiactive, Lunaequivalenceadaptersunderreview.

## Current authority — 2026-09-30 11:25:36.838 UTC checkpoint

- Bonsai **110507** remains active at four concurrent requests; sampled GPU **97%**. Hourly journal audit: **138 completed finishes, zero non-success finishes** since10:24; current queue has **zero future unstarted held rows**. Next full inspection **12:25:36.838 UTC**.
- Luna workers186820/186821 completed all13 title variants with exact output-accounting/2 proof. **No live Luna generation worker.** One raw accepted;12 rejected and held. Initial review finds mostly source-supported wording/specificity differences. Do not launch another title retry batch until bounded evidence-backed equivalence proposals are root-reviewed; no broad fuzzy comparator, no automatic DPO negatives, no numerical value relaxation.
- Current static **quality-v22**, fourbundles **6,150 cases /32,667 approved decisions**; directory1,224cases/3,227decisions. Nine added holds sincequality19, originals preserved. `current_publication_revision` now identifies current publication; inherited `quality_revision` describes older origin.
- **Recipe-v22 published**, separate failure inventory and all required source lanes included; fresh exact carryforward/hash/readiness audit underway. Recipe21 and its audit retained as historical/stale. No final student render/split/token/dedup or training.
- Authoritative health JSON: `runs/generation-check-20260930-hourly/hourly-health-next/health-2026-09-30T112536838Z.json` (SHA0931ae3b…523c2534). Earlier11:24:50 report has per-file async-iterator read errors; preserved, NOT evidence for zero generated outputs. Reader corrected. These direct queue output counts do not include all Bonsai per-batch saved/spooled jobs; canonical generation output supplement is underway.
- check.json flat quality/recipe fields are reconciled; next inspection follows its clock. State records completed workers; never restart exhausted queues. Disk sample9.6GiB free; gzip future snapshots only. No tests/studenttraining run.

## Live generation update — 2026-09-30 11:16:36 UTC

- Two Luna workers **186820 / 186821** now run **13 independently source-reviewed MuSiQue title-path variants**, split 6/7, frozen runtime-v19, one request each, 40 turns / 256 requests / 1,200 seconds and existing backoff. Authority: `runs/generation-check-20260930-hourly/musique-title-path-r2-next/revision-4/launch.json`. Previous 24/21/17 proposals are preserved and must not be launched.
- Bonsai **110507** continues with four requests. Next full check remains **11:24:31.182 UTC**. No exhausted queue restart or frozen runtime mutation.
- Failure candidates now have a separate content-addressed compressed inventory, full original evidence, manifest and recipe carryforward receipt (commit `e599ca7`). Default command stdout compatibility preserved; recipe opts into JSON. No negative labels assigned. Catalog now schema-dispatches failure versus positive manifests; first refresh exposed/fixed a KeyError, interrupted artifacts preserved.
- Recipe-v21 published with quality-v20 sources and failure inventory; subsequent holds make it stale. Refresh after the next hourly check and settled quality-v22. Do not report snapshot selection as final training readiness.
- New source review uncovered nine additional holds since quality-v19; final quality-v22 publication is underway. Two further relation ambiguities (Esperanto formation city; Egypt century qualifier) remain review and excluded from this queue. Root withdrew incorrect flags: explicit Unicode subset supports WGL-4, military branch of XXX Corps means British Army, and a proportional plague answer need not be a headcount. Root corrected Mongol/Jin support to the actual Yuan article.
- Route audit confirms child calls: **2,566 approved child decisions**, 5.34% of 48,033 audited static-plus-generated projection decisions; this excludes legacy/general/coding/final split/dedup. Nine selected rows fail materialization (seven legacy checkpoints; two missing invocation attribution). Keep excluded; no fabricated trace migration. Reports: `runs/data-lineage-20260930/route-audit/`.

## Current authority — 2026-09-30 10:52 UTC

- Bonsai PID **110507** is running frozen runtime-v19 with four concurrent requests/server slots. Its last journal completion was followed by a new start; generation is progressing. This process imported the older supervisor; use output-accounting/2 at the next natural campaign boundary.
- Both Luna source-derived jobs finished successfully: clarified SciFact claim returned SUPPORT (2 admitted decisions), explicit MuSiQue year question returned 380 (5). These are new cases with new teacher runs; the malformed originals remain held. No Luna generation worker is currently live. Preparing two queues of up to 12 source-qualified title-path cases each; review before launch.
- Last full health checkpoint: **10:24:31.182 UTC**; next due **11:24:31.182 UTC**. Evidence: `runs/generation-check-20260930-hourly/hourly-health-1017/health-2026-09-30T102431182Z.json`. Disk free now about **9.7 GiB**; preserve originals and use compressed snapshots.
- Four current static bundles contain **6,159 cases / 32,676 approved decision IDs**. Directory quality-v19 has 73 cumulative holds. Recipe-v20 is stale: refresh after the pending failure-inventory integration, carrying all four static bundles, both legacy reference sets, approved minimum-age migration and the two new teachers.
- Joint training now requires a hash-bound version-2 reducer mix audit and current policy/inventory readiness; checkpoint identity includes gate hashes (commit `bd7bffd`). **25% is a minimum admission gate**, not an exact sampling share. Final student rendering, token budgets, source-group splits and deduplication remain required; no training has run.
- Exact numerical computation remains exact. The reviewed minimum-age migration accepts equivalent expressions of 21 only, with wrong ages rejected. No numerical target injection or general tolerance is enabled.
- DPO audit found rejected attempts are omitted from the positive-only snapshot and automatic pair discovery. A separate durable failure-candidate lane is being implemented. Candidates require approved repairs and existing same-prompt, current-runtime causal checks before becoming pairs; source-held outcomes are not negatives.
- Delegation distribution audit is still underway. Counting explicit top-level delegate calls alone misses child NL calls inside evaluation; do not report zero delegation from that measure.

### Historical updates (newest first; current authority above supersedes stale counts)

> **2026-09-30 10:17 UTC current policy/accounting.** Supervisor output-accounting/2
> rootapproved in scripts/run_bonsai_queue.py; newinvocations validateexactnative
>embeddedIR/exportrows/manifest/sourcehash beforecomplete; explicitpolicy skips
>normalexitsonly, hardtimeouts/inactivitypreserved. Bonsai110507 stillold imported
>runner untilnextnaturalboundary/campaign; no urgentrestart. Savedaccepted/rejected
>exports pass, old15turnemptyexport fails; originalsretained. No tests.
> Originalmissing MuSiQue question asksgroup, gold380year; centrallysourceheld.
> Staticquality-v19:1233cases/3236approved,73cumulativeholds, catalogcurrent.
> Recipe20 staleby1hold, refresh21afterhourlyaudit. Fresh distinct WHEN derivative
>+clarifiedSciFact prediction derivative beingpreparedfortwoexistingLunaslots.
> Scheduled10:17:18health assignedLunaadmissionagent; currentauthorities unchanged.

> **2026-09-30T10:11:47.549Z minimum-age contract approved.** Futurebuilder applies
> exact evidence/full-base-digest pinned minimum-age adapter afteralias/groupassembly.
> Canonical+titled variant controls rejectgold/prompt/source/ref/groupdrift. Root
> approved3decision migration of unchanged teacher answer21yearsold; raworiginal
> remainsstrictrejected and preserved. New separatelyapproved .result at
> hourly/musique-minimum-age-reviewed/audit-revision-2/migrated-saved-pilot.result.json;
> root-approval.json pinsbothhashes. Earliercandidatecopy supersededbylatestapproved.
> Recipe20 predatesit; nextrefresh carriesitautomatically. Noprompt/answerinjection.
> Future review-only candidates MUST use .review-candidate.json, not .result.json,
> toprevent auto snapshot admissionbefore rootapproval. No generalfuzzyoracle.

> **2026-09-30T10:06:13.215Z recipe-v20/current policy.** Snapshot2542selected
> teachertrajectories/2075programs, Luna1774/Bonsai634; explicit792+1400legacy
> andallfourstaticincluded. Staticquality-v18:1234cases/3237approved;72cumulative
>holds; fourstatic32677approved/6160cases. No missingdefaults/qualityblockers.
> Gzip fullsnapshot hashes independently verified:67344798compressedbytes vs
>523126712uncompressedbytes (~64.2MiB vs498.9MiB). Earlier~2.5GiB claim was
>diskdelta estimate, not this serializedsnapshot size; use measuredbytes.
> Uncompressed/content SHA2ccdc...b1de; compressed32988...39b8; manifest/2.
> Currentrecipe20 not finalstudentrender/split/token/dedup/25%mix approval; no training.
> MuSiQue11consolidatedholds publishedroot-fifth-batch; prior pendingfifth superseded.
> SciFact694original held for Lice/Live AND unqualifiedstronger/causalclaim; typo
>alone insufficient to admit gold. Source-derived narrower newcase beingprepared,
>not a trajectory migration. Minimum-age adapter rootreview ongoing; no builderwiring.
> Supervisor empty-success accounting fix assigned; no liveprocesspatch yet.

> **Luna next304 completion audit:**302actual joinedrows,155rawaccepted/147rawreject,
>150current-native candidates. MuSiQue246rows:101rawaccepted/145rawreject,96candidates;
> SciFact56rows:54accepted/2reject,54candidates. These152admissionheldrows are NOT
>152bad source roots; candidate/migration/source reasons differ. Two missingrows
>preexisting: source164 originalfalse-inactivity(known reviewed retry retained),
>source3hop1__11265_35341_79479 empty export despite complete journal underreview.
> Since09:17 no noncomplete finishes. SciFact480 clearsemanticpolarityerror;
>694possible Lice/Live typo underreview. Evidence luna-next-completion-audit-0944.
> Five root source-read holds staged root-pending-fifth-batch.json, NOT published:
>Swedish official-versus-standard, Dill header/Brockdalebody, unnamedmultiple
>Thessaloniki churches, unspecified Astros playoffopponents, Johnson township
>Indiana vsNewYorkcountychain. No training; consolidate before recipe refresh.

> **2026-09-30T09:48:47.005791+00:00 pilots completed.** Both isolated queues done normally:
> Islamicmathematics14.9s/2replies admitted2decisions; Iowa23.7s/3replies returned
> 21 years old vs minimum-age gold21orolder. Source-supported equivalence held,
> not modelnegative; root reviewing pinned minimum-age adapter. Each read only
> two title-selected articles. No claim measured improvement vs matched control.
> PIDs133004/133005 no longer live; check/state completed. Bonsai110507 continues.
> No exhausted queue restart. Original results retained; no target injection.

> **2026-09-30T09:46:46.851176+00:00 two Luna pilots launched.** PIDs133004/133005,
> isolated pilot-v19 queues/journals/jobs; gpt-6-luna low cap1 each,1200s/40turn/
>256requests/executionplans/backoff unchanged. Native title-path-r2 variants retain
> golds/oracles/sourceproofs/bodybytes; first-lineH1 and stale reference path controls
> fail closed. Root checked hashes, idempotence and current source eligibility.
> Helper standalone; no sharedbuilder wiring. Launch authority
> hourly/musique-title-path-v2/pilot-v19/launch.json; check+state updated. Old completed
> queues/journals retained. Bonsai110507 v19 cap4 ongoing. Nextfullcheck10:17:18UTC.

> **2026-09-30T09:43:36.988517+00:00 source/storage update.** Full included MuSiQue audit282/282
> completed,198pass/84review flags; question-consistency correction underway.
> Flags are not holds. Root confirmed four more cases: Ålanders wished but never
> rejoined Sweden; separate Law & Order series; two song-subject/video-rumor chains.
> Central + staticquality-v16:1246cases/3249approved,60cumulativeholds. Catalog
> replacements recorded, recipe19 now stale; no training. Frozenv19 untouched.
> Both Luna queues completed normally through index151; oldPIDs111689/110834
> absent, state queue_completed. Bonsai110507 live. Two title-path pilot variants
> being prepared for existing two Luna slots; do not restart exhausted queues.
> Future snapshots now gzip, stream readers plain/gzip, compressed/content hashes,
> fsynced data+ledger and atomic manifest-last publication. Tiny actual native-row
> admission/materialization equivalence and bad-gzip fail-closed audited; syntax
> passes. No fullsnapshot yet; preserve all old artifacts. About13GiB free.
> Title filename helper remains standalone pending rootreview, no builder wiring.
> Next scheduled full health10:17:18UTC; continue active hourly loop.

> **2026-09-30T09:22:07.154305+00:00 hourly check/recipe current.**
> Exactlive Luna111689/110834+Bonsai110507/v19 confirmed. Since08:13:19 zero
> noncomplete finishes. Nextpool246unique digestjoined results:102rawaccepted/
>144rawreject,98current-native candidates;20sourceholds,284eligible targetroots.
>66held queueoccurrences allfinished, zero unstarted heldroots: no urgentrollout.
> Root09:14GPU98%,serverRSS3.03GiB/container2.97GiBof6GiB; later09:17GPU0%
> instantaneous sample recorded, not a utilizationaverage. Diskabout15–16GiBfree.
> Recipe-v19 currentcentralpolicy:2473selectedgeneratedtrajectories/2020programs,
> fourstatic32693approveddecisions/6176cases; explicitlegacy792+1400refs retained,
> no missingdefaults/included explicitqualityblockers. Sourcegoldaudit ongoing;
> finalstudentrender/token/split/dedup/25%mix audits pending; no training/tests.
> Evidence hourly/hourly-health-0913/report-2026-09-30T091718031Z.{json,md},
> check-0914-system.json. Nextfullcheck10:17:18UTC; continue activehourlyloop.

> **2026-09-30T09:12:18.619166+00:00 source holds protected.** Eleven root-adjudicated holds
> now central policy + static directoryquality-v15:1250cases/3253approved,56
> cumulativeholds; fourstatic32693approved. Catalog replacements explicit. Raw
> originals/golds preserved; no modelnegative. Pending-third-batch superseded
> root-third-batch.json. Build passed. Teachers remain immutablev19; futurequeue
> check pending, consolidate next rollout after fullsourceaudit. Recipe18 stale.

> **2026-09-30 09:03 UTC retry admission/source audit:** TreeDST v19 fresh retry
> independently admitted, exact expected Tree, five materialized decisions/zero
> unlinked; only successful edit and finish approved. State+utterance visible in
> all decisions. Audit adjacent retry/outcomes.audit.{json,md}. Original timeout
> preserved/no DPO pair from runtime artifact. Full MuSiQue audit177/282 so far,
>121pass/56review; review flags require root adjudication, not automatic holds.
> Eleven additional root-reviewed holds saved pending-third-batch.json for next
> consolidated publication/runtime batch; no training, recipe18 remains stale.

> **2026-09-30 08:56:53 UTC all teachers runtime-v19.** Boundary watcher110320
> completed, no overlapping collectors. Luna111689/110834, Bonsai110507; immutable
> prep-v2 queues/journals/caps unchanged. Current state/check authorities updated.
> Saved TreeDST retry rawaccepted/allchecks in135.9s using omittedvaluefinish;
> final native admission audit pending. Luna slowcase101 finished726.5s/111replies:
> source asks salt lawBelgium but only India evidence; actual NOT FOUND is not
> modelnegative. Root added it to pending sourcehold batch, gold preserved.
> Next fullcheck09:13:19UTC; recipe18 stale after staticv14 sourceholds.

> **2026-09-30T08:55:07.855679+00:00 alias adapter reviewed.** Future MuSiQue builder uses
> reviewed-oracle-alias-removal-v2 for exact two bad source alternates. Raw originals
> preserved; three matching accepted legacy rows returned valid primaries, so no
> blanket hold. Native variants pass primary/reject removedalias, drift/idempotence
> audited. Runtime-v19 immutable unaffected. Bonsai110507 retry completed135.9s,
> correct Tree/all outcomechecks, used omitted-value finish; admission audit pending.
> Luna2 now110834/v19; Luna1 still100243/v18 waiting boundary. State actualpath
> hourly/runtime-v19-rollout-v2-state.json; watcher110320, do not restart.

> **2026-09-30 08:48:57 UTC v19 rollout launched.** Sole watcher110320 uses
> runtime-v19-rollout-config-v2.json, frozenmanifest5f0a9e...verified605files.
> Old v18 authorities100243/100068/100628 until respective journaled boundaries.
> Completed/current entries preserved; original held roots alreadyfinished, no
> future-held removals necessary. Independent TreeDST633ad retry first Bonsai
> item, original40turn/384request/1200s budgets, own source/jobs/output.
> Read state for replacement PIDs before further changes; never restart watchers.

> **2026-09-30T08:45:23.109715+00:00 pending quality work:**
> Five additional root-reviewed candidate holds saved in
> hourly/musique-included-source-sweep/root-pending-third-batch.json; deferred
> to next consolidated policy batch. No training; recipe-v18 remains stale.
> Two invalid source aliases (Islam; Gun laws in Iowa) found; no accepted recipe
> rows returned those aliases, all three matching rows returned valid primaries.
> Evidence-scoped modern alias-removal adapters assigned Luna sourceoracleagent.
> v19 rollout preparation first draft superseded: retry budget/order and
> completed/current grouped-entry preservation must be corrected before launch.

> **CURRENT — 2026-09-30T08:37:15.270331+00:00 source-policy/finalization batch.**
> Six independently reviewed MuSiQue source holds published as directory quality-v14:
>1261 cases/3264 approved decisions,45 cumulative holds; all four static32704.
> Original golds/results preserved; no preference negatives. Recipe-v18 is STALE
> relative to these holds; consolidated refresh pending next health/audit batch.
> Staged-result omitted-value finish fix built/native-audited, contract19; runtime-v19
> immutable queues/boundary rollout being prepared, NOT launched yet. Existing
> v18 authority remains Luna100243/100068, Bonsai100628. Next full09:13:19UTC.
> Full included MuSiQue source audit ongoing; root must verify flags before holds.

> **CURRENT — 2026-09-30 08:16 UTC hourly check/inventory complete.**
> All v18 workers healthy: Luna100243/100068, Bonsai100628; GPU99%,7457/8188MiB,
> server35762/container2.295GiBof6GiB. Disk18.23GiB before latestsnapshot. Fullcheck
> actual08:13:19UTC saved check-0812.json; next09:13:19UTC. Nextpool106unique
> results:36rawaccepted/70reject, MuSiQue sofar; rawacceptance not finaladmission.
> Only new noncomplete events: known fixedfalseinactivity0012 (independentretry
> finished) and Bonsai broader-expansion346retained1 genuinehard1200stimeout,
>11replies/23558tokens; savedtrace investigation assigned Luna sourceoracleagent
> hourly/timeout-0813-bonsai. No other new transport failures. Whole282included
> MuSiQue source-quality sweep assigned Luna qualityagent under hourly/musique-
> included-source-sweep; root must fullread all flagged cases beforeholds.
> Recipe-v18 nowcurrentpolicy:2362selectedgeneratedtrajectories/1931programs,
> allfourstatic32710approved, no missingdefaultinputs/includedqualityblockers.
> Legacyref inputs remain explicitly included. Finalstudentrender/split/token/
> dedup/25%mix audits remainpending; no tests/studenttraining. Continueactive
> threadhourlyloop, do not restart completedwatchers or mutate frozen runtimes.

> **CURRENT — 2026-09-30 08:06 UTC: all teachers runtime-v18.**
> Boundary rollout100001 complete, no overlapping collectors. Luna100243/100068,
> Bonsai100628, same queues/journals/caps/server. Exactnumeric comparison and six
> sourceholds deployed. Nextfullcheck08:11UTC then consolidated recipe refresh.

> **2026-09-30 08:05 UTC corrected precision audit:** exact recipe-v16 ordered
> existing_teacher_results plus all four current static manifests audited:10691
> saved resultrows,322 accepted generic-normalized answerchecks,0changedpositives,
>0QAspanchecks. Includes792ref-v1/1400ref-composed/2306snapshot/6193staticcases.
> Old1278-subset audit preserved/superseded withhash; root verified exact paths/counts.
> Frozenv18 matches current built tree. Luna100243/100068 nowv18; Bonsai97277 waits
> boundary on broader-expansion346 retainedgroup; check state for replacement PID.
> Next full health08:11UTC, recipe refresh once per hour/batch.

> **CURRENT — 2026-09-30 08:02 UTC: v18 numeric precision/source holds rollout pending.**
> Frozenv18 adds exact decimal comparison to generic normalized answers and QA
> numeric guards, without generic rounding. Collector answer_comparison_version
> normalized-decimal-exact/2 forbids stale checkpoint/result reuse for new collection.
> Root found float normalization could equate distinct large integers/highprecision
> decimals. Luna is correcting its data audit: initial scan mislabeled an older
>1278 snapshot as recipe-v16; exact recipe input is537e372... with2306 trajectories.
> Do not claim current snapshot audited until corrected report/identity joins finish.
> Six root-reviewed MuSiQue holds from41-result nextpool audit: next0/154/159/165/
>23/158 for date/count, wrongbodyterm, pluralgold, metric, age/scope, wrongeventdate.
> Preserve all golds. Next4/169 flags withdrawn after full root reads: Scottish1698
> and LadyGagaFame chain are supported. Static directoryquality-v13 now1267cases/
>3270approved/39cumulativeholds, allfourstatic32710approved, catalogchain explicit.
> All530 eligible original file/tree cases have model results, zero missingeligible.
> Including v12/v17 retries489 have rawacceptedattempt: C287/287,TATQA183/222,
> TreeDST19/21. Rawacceptance is not current/finaladmission. Reviewedlakhvariant
> separate. Four v17 fresh retries pass (threeMarkdown and actual full-datealias);
> infrastructure retry finishes with source-supported expandedKhagan answer, held
> for extractiveequivalence instead of modelnegative. Evidence original-cohort-
> coverage-0758.json and reviewed-retries-v17/outcomes.audit.json. v18 solewatcher
> migrates Luna97137/97063 and Bonsai97277 at boundaries, same queues/caps/journals.
> Nextfullhourcheck08:11UTC and consolidated recipe refresh; no tests/training.

> **CURRENT — 2026-09-30 07:45 UTC: all teachers runtime-v17.**
> Boundary rollout complete, no overlapping collectors. Luna97137/97063 on
> combined v17 queues; Bonsai97277 on existing queue-v44. Same journals/caps/server.
> Source-specific supervisor monitoring active; five independent retries ongoing.
> Next hourly inspection08:11UTC; refresh recipe once after current quality batch.

> **CURRENT — 2026-09-30 07:43 UTC: v17 boundary rollout pending.**
> All v16 workers migrated normally: Luna93802/93803, Bonsai94895. Sole new
> runtime-v17-rollout watcher waits for their journaled boundaries, preserving
> queues/caps/journals. Check state for new PIDs before further changes. Five
> retries precede Luna continuation: CommitPack136/180/73 scoped Markdown variants,
> reviewed MuSiQue date alias, independent retry of false inactivity interruption.
> Supervisor bug: original and next campaigns reuse indices in jobs directory;
> old index-only lookup chose an unrelated completed result over current partial.
> Interrupted worker2 next0012 actually saved45 replies, old telemetry reported5.
> Fixed lookup uses frozen collector digest plus program ID/provenance hash, picks
> latest matching result/partial once per root; retry deadlines also source-specific.
> Raw artifacts/journal timeout retained; no false model-negative label. Evidence
> hourly/supervisor-program-identity/audit.json. Existing old PIDs only adopt fix
> when restarted at boundary. No change to hard/inactivity budgets or backoff.
> Markdown comparator is opt-in for changed allowlisted .md target only: remove at
> most one EOF LF/CRLF; body, other files, code files remain exact. Missing targets,
> extra blanklines and unrelated edits fail; EOF-only tasks keep old exact contract.
> Future CommitPack variants clearly request replacement exactly once. Native
> references73/136/180 pass; old136/180 migration holds, duplicate73/truncated42
> remain wrong. MuSiQue date alias pins input hashes; old trace not promoted.
> Recipe-v16 predates new migration classifier; refresh once with next hourly batch.
> Native/build audits only, no tests or student training. Next full check08:11UTC.

> **2026-09-30 07:32 UTC inventory refresh:** recipe-v16 now includes current source holds,
> all four static bundles (32716 approved decisions), 2306 selected generated
> trajectories /1881 programs; no missing default inputs or included quality blockers.
> These are admitted candidates before final student render/split/token/dedup/mix
> audits, not training already run. Luna93802/93803 remain v16; Bonsai87740 finishes
> its v14 boundary case. No numerical target injection enabled; exact-input code
> and computational results must remain exact, as the user explicitly confirmed.

> **CURRENT — 2026-09-30 07:20 UTC: Luna extension active; Bonsai v16 boundary pending.**
> Runtime-v15 was prepared but never deployed. Sole rollout93792 moves v14 DIRECT
> to frozenv16. Original Luna queues exhausted normally; new Luna PIDs93802/93803
> are on v16 combined queues, retaining all historical keys/journals. Bonsai87740/v14
> finishes case91 then moves to queue-v44/v16. Check latest v16 state for exact PID.
> Same caps: two Luna provider workers, four Bonsai/server slots, exponential backoff.
> Next pool304:248MuSiQue/56SciFact,152per Luna worker. Root verified hashes, every
> queue index/key and current/frozen eligibility.301 future duplicate Bonsai roots
> reserved to Luna; completed/current attempts preserved, clean contiguous groups
> retained. Three other selected sources were already absent/completed in future
> Bonsai range. QASPER300 remains generation-held, verified static rows included.
> Next-pool prior coverage121 admitted successes,18 explicit failed-attempt repair
> backlog; static references not counted as generated. Manifest/prior-job-coverage
> in runs/luna-directory-next-20260930. No source/gold/reference/group rewrites.
> Full TATQA sweep239 rows: agent222pass/16review/1existinghold. Root checked ALL16
> review source inputs before applying holds. Some agent IDs/content/calculations
> needed correction; do not trust ordinal-based recommendations. Root classifications
> override EPS wrong-sign assertion (direction ambiguity), goodwill wrong-aggregation
> assertion (population/dash ambiguity), and other unsupported flags withdrawn.
> Directory quality-v12:1273cases/3276approved/33 cumulativeholds; TATQA222 remaining.
> Allfourstatic32716approved.16 new source holds preserve all old golds/traces and
> static publications, no auto DPO negatives. Catalog replacement chain explicit.
> Original file/tree559 now530 eligible originals plus1 reviewedlakhvariant.
> Hourlycheck at07:11UTC: all workers healthy, GPU98%, disk19.86GiB;548 original
> origins had raw model results. Current recipe-v14 predates16holds; refresh next
> as one batch. Source snapshots use6.2GiB: avoid copying full snapshots per tiny
> change; publish policy/ledger immediately and consolidate inventory refreshes.
> New non-TATQA rejection audit hourly/rejection-audit-0708-other: README duplicate
> insertion is real error; two terminal-newline-only mismatches need narrow review.
> Nextfullhourcheck08:11UTC. Continue active thread loop; no student training/tests.

> **2026-09-30 07:00 UTC user numerical policy:** bounded synthetic exact-target correction may
> be considered for subjective judgment scores, with raw output/provenance retained
> and causal consistency reviewed. Code/computations from exact inputs stay exact
> (explicit rounding/format equivalence allowed). No injection enabled; current
> failures are arithmetic/units/scope. Returned actual file/edit counts and sums must
> agree with decisions/actions even when classification involves judgment.

> **2026-09-30 06:49 UTC audit workflow:** before applying source holds from a `new_cases`
> report, run `python3 scripts/verify_source_audit_identities.py AUDIT --output NEW`
> from repository root. It joins exact source/program/question identities to saved
> result.task.program_ir, pins both hashes, exits nonzero for mismatches, and writes
> immutable evidence. This checks identity only, not semantic correctness. Actual
> 0625 audit reproduced1 mismatch in identity-check-initial.json; existing Luna agent
> now correcting report/derivation description with backups and rerunning the check.
> Four broader-sweep root-reviewed candidates recorded pending next batch in
> hourly/tatqa-remaining-source-sweep/root-adjudications-pending.json. Current policies
> stillv14; final training release must wait for quality adjudication, no training run.

> **CURRENT — 2026-09-30 06:44 UTC: all teachers frozen runtime-v14.**
> Boundary rollout87674 completed: Luna87754/87694, Bonsai87740 queue-v42, same
> caps/provider/backoff/journals/server. Exact authorities updated. Recipe-v14:
>2192 selected generated trajectories/1852 programs, allfourstatic32832 approved,
> no missing defaultinputs or included quality blockers. Sourceidentity corrected;
> final student render/split/token/dedup and25%mix audits stillpending. BroaderLuna
> sourceaudit continues with explicit coverage; nextfullhourcheck07:08UTC.

> **CURRENT — 2026-09-30 06:42 UTC: expense case identity corrected; v14 rollout pending.**
> Earlier audit typed cc42e86a-c56c..., but saved IR is cc42e86a-c56d... . Root's
> source-policy join caught eligible547 instead of546; broader Luna sweep caught
> the same mismatch. Exact actual ID now centrally held. All other4 audit IDs
> matched artifacts. Agent correcting report with backup. Identity proof:
> hourly/tatqa-numeric-campaign/current-source-admission.audit.json. This corrects
> the earlier statement that the fourth case was absent from canonical static;
> it WAS present but failed to match the typo. Static quality-v9 now excludes it:
>1289cases/3392approved/17 cumulativeholds, allfourstatic32832approved. Golds/raw
> traces unchanged, replacement catalog explicit. Originalcampaign546eligible:
>287CommitPack/238TATQA/21TreeDST, plus1 distinct reviewedlakhvariant.
> Frozenv14 includes correctedID. Soleboundary watcher87674/config/state/log
> hourly/runtime-v14-rollout*; previousv13workers86353/86475/87000 until boundary.
> Future queues Luna-N.v14/ Bonsaiqueue-v42. No caps/journal/server changes.
> Recipe-v13 predates this correction; refreshv14 after publication. Broaderaudit
> in progress, nextfullcheck07:08UTC. Don't infer completeness from referencepass.

> **CURRENT — 2026-09-30 06:38 UTC: all teachers frozen runtime-v13.**
> Boundary rollout86322 completed: Luna86353/86475, Bonsai87000 queue-v41. Same caps
> (Luna2 provider, Bonsai4/GPU/server4), backoff and journals. Authorities updated.
> Recipe-v13 completed: 2156 selected generated trajectories / 1833 programs;
> all four static bundles present, 32837 approved static decisions, no missing
> default inputs or included quality blockers. Final student audits pending.
> Broader read-only included-TATQA source audit assigned to existing Luna quality
> agent: hourly/tatqa-remaining-source-sweep. No extra provider/GPU jobs. Root must
> check evidence before any hold/recovery. Next full hourly check07:08UTC.

> **06:34 UTC: four more source holds, v13 boundary rollout pending.**
> Root checked full tables, notes and references from the 0625 numeric delta audit.
> cc42e86a: signed expense table versus positive expense-magnitude note/gold;
> 122bddf9: gold uses OTHER ACCOUNTS for both years despite question asking Costs
> and Expenses (correct requested-column change is1). Agent report initially said
> mixed columns; correction requested. 3257598a: no visible million unit for equity;
> 85d145d7: 2019 question versus two-year average gold. Hold all four, preserve raw
> gold/inputs/traces; no automatic relabel or DPO negatives. 5688b0ff remains genuine
> totals-row model error. Source review compiled/frozenv13, exact boundary watcher
> 86322 owns all teachers; read hourly/runtime-v13-rollout-state.json for new PIDs.
> Old v12 PIDs84926/85014/85050 until their boundaries. Same caps/provider/backoff.
> Future Bonsai queue-v41 retains clean members of filtered batches; new Luna v13
> queues preserve completed history. Campaign546 eligible originals plus1 reviewed
> lakh variant. Directorystaticquality-v8 removes3 published cases:1290/3397 approved;
> cumulative16 static holds, allfourstatic32837 approved. Fourth new held source was
> not in the current canonical static bundle. Replacement chain explicit in catalog.
> Recipe-v12 now predates these4holds; refreshv13 after rollout. Next07:08UTC check.

> **06:28 UTC results:** reviewed lakh retry obeyed the new unit format but still
> failed semantically (1602.85 included commission/allowances, expected gross-salary
> subtotal 242.5); preserved files, no scope/tool failures. Remains rejected. README
> retry accepted with complete expected edited file, including unrelated lines.
> No automatic DPO pairs. Outcomes hourly/reviewed-retries-v12/outcomes.audit.json.
> Recipe-v12 completed: 2116 selected trajectories / 1817 programs, all four
> static bundles present (32856 approved decisions), no missing default inputs or
> included quality blockers. Snapshot predates the final retry results; next refresh
> will discover them. Final student render/split/token/dedup/25% mix audits pending.

> **CURRENT — 2026-09-30 06:25 UTC: all three teachers on frozen runtime-v12.**
> Boundary rollout 84904 completed without overlapping collectors: Luna 84926/85014
> (two provider workers, cap 2, exponential backoff); Bonsai 85050, queue-v40, cap 4,
> server 4. Exact authority: hourly/runtime-v12-rollout-state.json, check.json and
> runs/luna-file-tree-20260930/state.json. Do not restart older rollout watchers.
> Latest three source holds now active in teachers. Original 559 file/tree roots
> remain preserved; 550 eligible originals plus one separately reviewed lakh variant.
> Lakh output bug fixed with explicit source-unit convention, original gold/inputs/
> reference unchanged. Strict shared validator pins source, hashes, gold, expected
> files, prompt and reference. Native replay: five turns, admitted, no tool failures.
> Original blocked trace remains held. Fresh independent Luna retries queued for
> this new variant and CommitPack README truncation; hourly/reviewed-retries-v12/
> manifest.json records preservation/hashes. Do not infer DPO pairs from retries.
> Audit: hourly/tatqa-lakh-reviewed-contract. Recipe-v12 refresh running (session
> 13333), includes directory quality-v7. No student training/eval or unit tests.
> Additional new numeric rejection audit assigned to existing Luna source agent:
> hourly/rejection-audit-0625-numeric. Next full check remains 07:08 UTC. Continue
> the active-thread hourly loop and log course changes; preserve frozen runtimes.

> **CURRENT — 2026-09-30 06:08 UTC hourly check.**
> Allteachersfrozenv11: Luna79716/79801, Bonsai80478 queue39 cap4/server4; GPU98%,
> disk23.31GiB. Freshprogress. Newfile/tree319uniqueoriginscompleted (265rawaccepted/
>54reject):CommitPack122/1,TATQA124/51,TreeDST19/2; numeric72/63accepted/9reject.
> Fullcheckhourly/check-0605.json; nextcheck07:08UTC incheck.json. Realnewpartial
> telemetry invocationID/timestamps/boundedtoolpreview/hashverified. Admissionallows
> direct/mixed/delegatedcorrectsolutions; stalecontrarycommentremovedonly, no rulechange.
> Directoryquality-v7 now1293cases/3416approved/13cumulativeholds; allstatic32856.
> Newholds87fe... absolutevspercentchange,813045... sign/direction, d3a... lakhformat.
> These sourceholdpatchesbuilt(notyetfrozen); v11teachingstillongoing. Oldraw/golds
> unchanged. Latestrecipe-v11 1964/1740 predatesnew3holds; refreshafterunitbatchdone.
> SYSTEM FORMAT issue: ourgenericpromptcan'texpressunlistedlakh despiteoriginalgold
> blank-scaleconvention. Lunaadmissionagentimplementingreviewedexplicitunitvariant
> sharedvalidator/adapter, outputhourly/tatqa-lakh-reviewed-contract. Rootmustreview
> before source-review.ts exception/freeze; don'toverwritedirtysource-review.ts.
> Newunitvariant mayexempt ONLY exactsource/hash/prompt/gold-approved d3case; old
> sourceblockedtracesremainheld/noDPOnegative/autopromotion. Newteacherfreshretry and
> audit-onlyreferenceifapproved. Keep ballmoving; no studenttraining/eval/unit tests.

> **CURRENT — 2026-09-30 05:55 UTC: all three teachers on frozenruntime-v11.**
> Soleboundaryrollout79671 completed: Bonsai80478/runtime-v11/queue39 cap4/server4;
> Luna79716/79801/runtime-v11/latestv11queues cap2/provider/noGPU/backoff. Supervisor
> authoritycheck.json and Luna file-tree/state.json updated. Old70991/v8rootended
> normallyat3600limit; finaljournalpreserved. CurrentBonsai nexteligiblecase47produces
> freshresponses/GPU93%; nooverlap/in-flightqueueedit/serverrestart. Alloldermonitors
> andruntime-v9/v10Bonsaistatesarchived/superseded; DO NOT restartthem. Readv11state.
> Latestrecipe-v11 completed1964selectedtrajectories/1740programs; fourstatic32879
> approveddecisions. Directoryquality-v6 has1296cases/3439approved/10cumulativeholds.
> LatestnewLunacampaign261completed/211rawaccepted;47newnumericvariants/41accepted,
>6rejects atprecheck (includes employeehold). Rawattempt counts differ from unique
> admittedsource decisions. Numericrejectionaudit delegated, hourly/rejection-audit-0604-
> numeric; awaitreport. Original559file/treesources retained;553eligible after6newholds.
> No source/goldrewrites or unsafeadmissionrelaxations. Allsourceunits/operation/tree
> audits and decisions below. Nextfullhourinspection~06:04UTC in hourly/check.json;
> keep active-threadhourlyloop; no studenttraining/eval/tests.

> **CURRENT — 2026-09-30 05:47 UTC: consolidated latest runtime rollout.**
> Quantityunit auditcomplete: employee7.2million isbadsourceunit; fiveotherreviewed
> share-countmagnitudeunitsaresupported. Goodwill123 versus61.5 remainswrongtotal
> insteadaverage; dashzero/populationambiguity noted, noextra hold. Finalsourceholds
> compiled/frozenruntime-v11. Directoryquality-v6/allfourstaticcounts3439/32879.
> Rootstopped ONLY pending obsoleteBonsaimonitors77278(v10waiting predecessor)
> and74250(v9waitingcurrentBonsai; itsLunarolloutsalreadycompletedandadvanced).
> Verifiedexactcommands/noBonsainewPID/no v10state; no teachers/server signaled.
> Supersessionproof hourly/runtime-v11-supersession.audit.json. Oldv9stateBonsaiwaiting
> isnowarchived/superseded; DO NOT restartv9/v10watchers oruse themas liveauthority.
> Newsoleboundarymonitor79671 owns allthree: Bonsai70991/v8/queue36 currentheld
> FOLIO425 finishesnormally, thenDIRECT→v11/queue39; skips intermediatev9/v10.
> Luna1already79716/v11/newluna-1.v11queue; Luna2still77384/v10 at recordedboundary
> (read latestv11state forreplacement). Samecap2/provider/backoff/journals; Bonsai4.
> Config/state/log/audit hourly/runtime-v11-rollout*. Sourcecampaignauthority
> runs/luna-file-tree-20260930/state.json perworker. Latestqueues omitthreenewholds
> andretaincleanmixedBonsaimembers; alloldIR/jobs/traces preserved, noin-flightedit.
> Neweligible553file/treeroots:287CommitPack/245TATQA/21TreeDST. Quantityfilteraudit
> hourly/tatqa-numeric-campaign/quantity-filter.audit.json. Recipe-v11 refreshsession
>4548, latestv10 snapshot1905/1704 predateslast3holds; updateoncecomplete.
> Continueactivehourlyloop; nextfullcheck~06:04UTC from hourly/check.json.

> **05:43 UTC source-quality update:** Directorystaticquality-v6 now1296cases/
>3439approveddecisions (10cumulativeheldcases); fourstatic32879approved. Three new
> sourceholds: c9026c61... headcount7.2 wrongly labeledmillion by mixedstaffcost table;
>8b50fee5... asksSanmina revenuechange but providesonlyshares;bd73aadd... combined
> two-datebalances versusindividualvalues. Rootfullinputsverified; originalgolds
> preserved, centralDPO/generation/admission holds. Independentquantity-unit audit
> ongoing at hourly/tatqa-quantity-unit-audit. Newsourceholdpatchbuilt, not frozen;
> batch into nextversionafteraudit, serializebehindexistingBonsai-v9/v10monitors.
> Currenteligiblefile/treetarget553(original559minus6newholds):287CommitPack,
>245TATQA,21TreeDST. Latestrecipe-v10 predateslatestthreeholds; refreshoncebatchdone.
> Chinesefoodleaf-casingaudit hourly/treedst-value-casing-audit confirmsonecaseonly;
> upstreamexactmatchpolicy strict, no globalcasefoldimplemented. Finaldataquality
> and25%mixstillpending. Continue activehourlyloop; nextfullinspection~06:04UTC.

> **05:32 UTC rollout/snapshot update:** Luna-v10 completed independently at
> boundaries: worker1PID77324,worker2PID77384 on numeric-campaign/luna-N.queue.jsonl,
> frozenv10; authority/check updated. First7newnumericcases all accepted (small sample,
> no rate claim). All249convertedLunacases verified original-arithmetic-gold, not year/
> count/span golds. Recipe-v10 completed1905trajectories/1704programs, no missing default
> inputs/included blockers; fourstatic32899decisions. Bonsai-v9 thenv10monitors still
> serialpending, check exactstates. Oldworkers not running; journals copiedhistory caveat
> remains. Operation/wording sweep delegated to luna_source_oracle_audit, readonly,
> hourly/tatqa-operation-contract-audit. Continue hourlyloop, nextfullcheck~06:04UTC.

> **CURRENT — 2026-09-30 05:29 UTC: numeric-contract rollout queued.**
> Frozenruntime-v10 adds three TATQA sourceholds and opt-in numeric-answer-v1 adapter/
> tatqa-answer-record oracle/files checks. Strict answer/scale strings, exact scale,
> numeric-only decimal/sign/grouped-comma/2dp equality; no prose/currency extraction,
> no fraction/percent conversion, half-cent ties exact-only. Return/file agreement now
> required. Four actual oldcandidate+native-reference audits pass, audit-only; old
> rawfailedtraces stay unchanged under legacy_tatqa_numeric_display_oracle migration
> hold(no falseDPOnegatives/auto-positive). Read hourly/tatqa-numeric-equivalence-review.
> FutureIR/queues in hourly/tatqa-numeric-campaign/manifest.json:559Lunarows249numeric
> updates; Bonsai1229IRrows255updates. Source/gold/groups/references preserved. New
> Lunaqueues retain historicalkeys except3explicitsourceholds(alreadyfinished). Jobdirs
> same; numericvariant IDs/checksums prevent incompatiblepartialreuse. Targetclean
> file/tree556cases(original559 minus3sourceholds); finalunique25%mix stillpending.
> Luna-v10 boundarymonitor77277 config/state/log/audit hourly/luna-v10-rollout* owns
> current74281/74312, samecap2/provider/backoff/journals, switches newqueue paths/v10.
> Bonsai-v10 monitor77278 waits runtime-v9 monitor74250 completion(all3runningstate),
> derives exactnewBonsaiPID onqueue37, then nextboundary→frozenv10/queue38. Current
> Bonsai70991/v8/queue36 still slowheldFOLIO425; preservecheckpoint/currentboundary.
> Read BOTH runtime-v9 and bonsai-v10 states before signaling. Never manually duplicate.
> Queue38 removes3futureTATQAholds and preserves6cleanmixedbatchmembers withsplitkeys.
> Source audit tablecopycorrection completed; allfourTATQAinputsverified,holdsstand.
> Staticdirectoryquality-v5:1299cases/3459approved; allfourstatic32899approved.
> Recipe-v10 refreshing(session45601); latestcompletedrecipe-v9 1851trajectories/
>1674programs(predates3TATQAholds). Fullnextcheck~06:04UTC in hourly/check.json.
> Keep hourlyloop active, inspect failures, refreshauthorityPIDs/queues afterrollouts.

> **05:16 UTC quality update:** Three TATQA aggregation/percentage-change cases now
> centrally sourceheld (e27c8621...,f6ef3a62...,7a6c059d...), golds retained. Directory
> staticquality-v5 published after hash/fulladmission verification:1299cases/3459approved
> decisions,7cumulativeheldcases; fourstatic32899approved. Catalog4→5links preserve
> history. Sourcehold patch built, not yet frozen/rolled; runtime-v9 rollout still
> owns currentBonsai70991/v8boundary; Luna74281/74312alreadyv9. Nextv10 must serialize
> Bonsai afterv9completion and freeze upcoming reviewed numericcontract if adopted.
> Latestrecipe-v9 predates threeholds; refresh recipe-v10 once sourcechanges settled.
> Numericrepresentation/precision audit delegated to luna_admission_dpo_audit at
> hourly/tatqa-numeric-equivalence-review; sourceagentcorrecting cross-case tabletext
> in hourly/tatqa-new-contract-review. Rootverifiedactual Industrial30/28net-salesinput.

> **05:11 UTC update:** 
> Recipe-v9 refresh completed: 1851 selected generated trajectories/1674 programs, no missing default inputs/included quality blockers; static approved decisions remain32935. Planning snapshot, not final training export.
> Luna runtime-v9 boundary rollout completed: worker1 PID74281, worker2 PID74312; authority/check updated. Bonsai70991 still awaiting currentcase boundary under monitor74250.
> 
> **CURRENT — 2026-09-30 05:09 UTC; supersedes older process details below.**
> Hourly check completed: Bonsai70991 frozenv8/queue36 cap4/server4/GPU99%; v7/v8
> rollouts both complete. Luna69180/69181 frozenv8, two provider workers/noGPU,
> new559file/tree campaign active; authority runs/luna-file-tree-20260930/state.json.
> At05:04 newcampaign96completed/69accepted/27rejected (raw runs, not unique admission).
> GenericTATQA scale retry accepted103.1/blank; MuSiQueWesternEurope accepted;
> NorthKorea-only retry remains wrong/incomplete. New rejection audit delegated to
> luna_admission_dpo_audit, output hourly/rejection-audit-0500 (await final evidence).
> Scoped child-lifetime fix built and frozen as hourly/runtime-v9, runtimecontract18:
> failed parallel eval settles its owned children before parent retry/environmentclose.
> Success with pending unawaited NL children now drains then errors before commit;
> external effects are not rolled back. Properly awaited parallel references172/182/212
> accepted, audit-only summaries hourly/scoped-child-drain-audit; no unit tests/GPU calls.
> Old slowpartial cannot faithfully replay; do not claim exact orphan-response counts.
> Partial journals now add callID/time/last-tool preview(max2000chars)/hash; hashes unchanged.
> FOLIOstory425 now source-review-held: hidden FOL conflates choosing/actual driving,
> visible English permits countermodels; original gold retained. No canonicalstatic rows
> affected. Root actual-range queue audit found six future425roots and one current root
> (agent initial queue count was incomplete). Queue37 removes only six future singles,
> preserves currentcase/other361entries. Audit hourly/folio-story425-source-review.
> Boundary rollout monitor74250 owns ALL three teachers, config/state/log/audit
> hourly/runtime-v9-rollout*. Each old exactPID/children must finish before replacement;
> same Luna queues/journals/caps, Bonsai switches queue37. Inspect monitor state before
> signaling anything; update newcampaignauthority/check.json once newPIDs launched.
> Recipe-v9 refresh running after hold; session52217 if still active. Latestprevious
> recipe-v8 has1676generated/1519programs; fourstaticbundles32935approved decisions.
> Planning only: no student training; final rendering/token/groups/dedup/25%-mix pending.
> Next hourly fullcheck ~06:04UTC in hourly/check.json. Continue active-thread loop,
> sleep <=60seconds chunks, investigate new rejections and keep decisions here logged.

# Handover: training data, preference pairs, directory reducers (2026-09-27 evening)

> **LUNA UPDATE — 2026-09-30 04:41 UTC: new file/tree campaign authority.**
> Workers69180/69181 now frozenruntime-v8, cap2/providerLuna/backoff/noGPU; previous
>61176/61177 both finished current roots and stopped before either replacement.
> Authority **runs/luna-file-tree-20260930/state.json**, monitor69128 completed.
> Queues retain each worker's entire balanced predecessor queue (finished keys skip;
> remaining Workflow roots finish first), then279/280new cost-balanced roots. Journals
> deliberately remain balanced/worker-N/journal.jsonl, including copied old histories:
> do not sum historical finish events across journals. Old result jobs retained in
> explicit worker directories; newfile/tree jobs in new worker directories.
> New559roots: CommitPack287visible-command file edits; TATQA251financial reducers with
> evidence-scale-v2 prompts; TreeDST21tree edits. Canonicalquality-v4source/hash in
> newcampaignmanifest; no source/evidence/gold/group changes. Six references accepted,
> admitted and matched expected. All287CommitPack exact-file requests audited: visible
> replacement defines edit, target/span provided, other files preserved; no hidden edits.
> No student training/GPUeval/tests. Target final25%unique admitted reducer decisions
> remains; raw case count/modality labels do not establish that finalshare.
> Bonsai54068/v6 still largeFOLIO212, nearing3600sec; v7monitor62418 thenv8monitor67297
> serialrollouts pending. Read source-aware audit hourly/folio-slow-212-audit when ready;
> freshdecode/repeatedwork does not prove useful semantic progress. Nextfullhourcheck
> remains hourly/check.json (~04:57UTC); keep monitoring/checkloop active.


> **UPDATE — 2026-09-30 04:07 UTC: TATQA unit visibility holds/publication.**
> Root verified3 additional defective scale contracts:13026d09... has million without
> visible scale;36308f38... has thousand for zero without visible scale;2b89071f... has
> blank despite explicit target in-millions header/note. Golds retained/source held;
> bc012d28... remains model-error (unsupported inferred million, no source hold).
> Full254-case audit hourly/tatqa-scale-visibility-audit. Directory static quality-v4
> published:1,302cases/3,495approved decisions/1,030existing excluded decisions;3 newly
> held,4 cumulative across revisions. Four static bundles32,935approved decisions.
> Catalog replacement chain3b→4 preserves prior data. Recipe-v8 planning snapshot
>1,676generated trajectories/1,519programs; no omissions/included blockers. Final
> student rendering/token/group/dedup/25%-mix still pending.
> Bonsai v7 monitor62418 still awaits current54068 boundary; **v8 monitor67297 waits
> for v7 rollout state to complete**, then resolves exact new PID and moves it at
> another case boundary onto frozenruntime-v8/queue-v36. Queue-v34 removes3 future
> defective TATQA roots, preserves9 clean roots from mixed batches and current root.
> Read BOTH monitor states before any manual action; neither overlaps collectors.
> No global scale prompt default added while source conventions remain disputed.
> Two reviewed MuSiQue aliases are implemented with exact primary gold, source snapshot
> and evidence-file hashes. New distinct-ID variants preserve visible prompts/golds.
> Four native audits accepted both primary/alias answers (audit-only, no training rows).
> Queue-v36 adds1 generic TATQA evidence-scale-v2 prompt retry then2 MuSiQue retries
> before v34; frozenv8 handles their
> standard oracle.alternates. Pending v8 monitor65189 was stopped before predecessor
> finished or any v8 worker launched, replaced by66207, then67297 after the generic TATQA variant was prepared. No v8
> worker had launched at either monitor replacement. Historical rejects stay held.
> Future TATQA adapter prompts clarify scale from question/evidence for requested
> quantity; empty for dimensionless/unstated scale, no inference from unrelated rows.
> New :evidence-scale-v2 ID/task_contract_revision preserves gold/groups/evidence.
> Actual native reference replay accepted; reference audit is not training data.


> **CURRENT — 2026-09-30 03:56 UTC; supersedes earlier process/snapshot details.**
> Bonsai54068 remains queue-v33/runtime-v6, cap4/server4, GPU98–99%. Boundary
> monitor62418 awaits current handoff:392:train-v32:0 before moving onto frozen
> runtime-v7; inspect hourly/bonsai-v7-rollout-state.json before signaling/restarting.
> No student training/GPU eval. V7 adds safe same-block positive const-literal counter
> steps (e.g. i += chunk with const chunk=3000), preserving mutable/dynamic/zero/
> negative/counter-shadow refusals. Saved MuSiQue snippet refused22 times now compiles.
> Audit hourly/const-step-audit/report.json; Node build passed. Collector records
> counter_loop_policy_version2 and requires it for checkpoint reuse. No unit tests.
>
> Luna balanced workers61176/61177, cap2/backoff/noGPU, unchanged frozen runtime-v5.
> **Authority runs/luna-directory-balanced-20260930/state.json.** Coordinated exact-PID
> boundary stop finished both prior cases before either restart.150 pending entries
> split75/75 by visible-input size; source/index/seed/jobs retained, outputs/logs now
> worker-specific. At03:52,144 remained across the new queues; old jobs directories
> remain authoritative result storage. Both new journals contain hashed copies of
> BOTH old histories for resume: never sum copied finish events across journals.
> Predecessor explicit state marked rolled-forward. New helper supports old_journal,
> a restart barrier, distinct new journals and filtering history by queue membership.
> Actual rollover audit runs/luna-directory-balanced-20260930/rollout.jsonl.
>
> Latest raw Luna chain361 completed artifacts/342 oracle-accepted (includes reviewed
> retries; NOT unique training examples). Current explicit results207/197. Three new
> rejects:199 missed unfulfilled work after malformed tool calls;201 ignored succeeded
> cancellation/reschedule;444 falsely blocked after incomplete bulk reading. Prior7
> mostly criterion confusion; no new schema failures or justified new source holds.
> Reports hourly/rejection-audit-0335 and -0355. Case135 asks policy-required handoff,
> not missing verification/tool503/voluntary actual handoff: root retains model-error
> classification despite initial subagent hold suggestion. claims_supported permits
> factual support in tool error payloads too; do not invent a successful-call-only rule.
>
> Bonsai11 completed entries last hour, no timeout. MuSiQue169/185 rejected plausible
> evidence-supported paraphrases. **New admission/DPO guard:** all MuSiQue answer
> mismatches require unreviewed_extractive_answer_equivalence (same as QASPER), with
> raw verdicts/golds retained. Accepted MuSiQue generation continues; reviewed aliases
> and faithful replay remain pending, no false-negative promotion. Actual5-row audit
> confirms2 held paraphrases/2 admitted exact positives/TATQA mismatch retained.
> TATQA176 computed103.1 correctly but invented million with no visible unit; root
> does NOT add a source hold just from that convention. Scale-visibility audit ongoing
> before any global prompt default (some other gold scales may lack explicit units).
>
> Planning-only recipe-v7 automatically includes1,638 selected generated trajectories/
>1,481 programs and all4 canonical static bundles (32,969 approved static decisions).
> No inventory omissions/included quality blockers; final student render/token/split/
> dedup/25%-reducer-mix still pending. Source disputes, evaluation and migration holds
> remain. Continue hourly active-thread checks: exact next due hourly/check.json.
> Frozen runtimes immutable; no hosted automation. Unrelated untracked research/plans
> and unsloth_compiled_cache must remain untouched.


> **CURRENT — 2026-09-30 02:35 UTC hourly check; supersedes all process/publication details below.**
> Bonsai54068 now runs queue-v33/runtime-v6, cap4/server4; rollout completed without
> overlap. Source holds excluded3 more future roots after reserved-seed filtering.
> Luna53036/53160 run runtime-v5 and explicit-field campaign; cap2/backoff/noGPU.
> Authority: runs/luna-directory-explicit-20260930/state.json. Its journals remain in
> the readable predecessor directories; prior states are marked rolled-forward.
> Pending499-case chain retains gold/source groups; one source-held case excluded.
>
> At start128 completed/120 accepted. Eight audits: five false schema blocks, one
> retrieval churn/late any(...) error, one rubric dispute, one genuine factual-support
> error. Fresh directory-v4 prompts list exactly the fixed filename-stem fields and
> state how to place them in return_result.value. Eval structural cutoffs now name
> a read_page route to the full value, starting page1; native transcript remains full.
> Eight references passed complete-input proof; actual large eval replay read10 pages.
> Seven reviewed retries: all six schema/retrieval blocks repaired; factual-support
> case77 still rejects (confuses quoted agent policy with tool-supported timelines).
> No automatic DPO labels from these retries; changed prompt/context must be respected.
>
> Source holds preserve gold: Workflow alias435c... claims_supported bank-advice
> ambiguity; FOLIO story395 exclusive/inclusive either-or ambiguity; TATQA directors
> chairman ambiguity (previous check). FOLIO's exclusive released premise can explain
> source gold; do not relabel the allegedly tautological conclusions. Audit reports
> in hourly/rejection-audit-0208, folio395-audit, static-holds-0208.
>
> **Canonical static publications filtered, original files retained:** Workflow
> quality-v3:4,803 cases/28,958 approved decisions,2 cases held. Directory-expansion
> quality-v3b:1,305 cases/3,529 approved decisions/1,030 already-excluded decisions,
>1 case held. Hold ledgers and prior manifests live beside each canonical manifest;
> validated all retained rows and hashes before publication. Other two bundles pass.
> Four bundles now32,969 approved decisions; old32,993 claim superseded. Failed first
> directory quality-v3 scratch is unpublished; canonical uses quality-v3b. The first
> filter attempt incorrectly assumed every materialized decision was approved; fixed
> to retain existing exclusion flags/counts. No source labels/input histories changed.
> New filter-static-bundle.mjs only permits explicit source/evaluation holds, otherwise
> fails closed. Replacement catalog now follows revision chains and rejects cycles.
>
> Latest planning-only recipe-v6 includes current publications and automatic snapshot:
>1,448 selected generated trajectories/1,291 programs, no inventory omissions/quality
> blockers. Not student-ready; render/token/split/dedup/mix remain. No student training.
> Exact next hourly due in hourly/check.json (~03:35 UTC, Berlin+2h); continue active
> sleep/check loop. Keep frozen v5/v6 immutable and re-read process states on resume.


> **CURRENT HOURLY CHECK — 2026-09-30 01:08 UTC, supersedes process details below.**
> Typed Luna batch64 completed:61 accepted,3 independently checked semantic errors,
> zero format failures. New readable typed directory-v3 adapter keeps full source
> instance mappings/golds and uses question names rather than opaque hash keys.
>499 fresh reducer cases prepared,8 reference replays accepted with complete visible
> inputs/no model calls. Two Luna workers45744/45745, cap2/backoff/noGPU, immutable
> runtime-v4; `runs/luna-directory-readable-20260930/state.json` owns queues/logs.
> Callback compiler now propagates declared Promise<boolean> slots through unannotated
> async callbacks; saved source-pattern analysis returns boolean/no diagnostics.
> Bonsai43115 still on runtime-v2, cap4. Boundary monitor46447 targets runtime-v4 and
> queue-v32, preserving current handoff:727. It excludes56 future reserved roots;
>18 clean roots from mixed batches retained as separate entries. Check monitor state
> before signaling any process; never run overlapping supervisors.
>
> **Breaking admission decision:** native s102 probe/s900 test source groups remain
> evaluation-only even when old artifacts say train. Audit found1,835 historical
> accepted teacher/train s102 trajectories (1,575 previously passed admission), no
> released scope. Central admission and downstream SFT/stage/DPO guards now protect
> these groups. New builder refuses training on reserved seeds. All raw data preserved.
> Refreshed generated snapshot selects1,278 trajectories/1,121 programs; raw2,775
> held artifact count is not unique examples. Snapshot manifest beefc689 under
> data/teacher/generated-snapshots; seed report under hourly/seed-audit. This decreases
> selected data intentionally; regenerate recipe/stages, never reuse old approval.
> TATQA source80fd3023-eb50-4db7-a299-ffcd06d604fa held: gold58.75 excludes the
> executive chairman, plausible answer60.4 includes him. Preserve gold, no DPO negative.
> Latest source hold/downstream script changes are in working runtime, after frozenv4;
> do not modify v4. Admission/export uses current code. No student training authorized.
> `runs/generation-check-20260930-hourly/check.json` stores exact next due time (~02:08
> UTC/04:08 Berlin). Continue requested hourly active-thread sleep/check loop.


> **HOURLY MONITORING RESUMED — 2026-09-29 23:50 UTC.** User requests continuing
> hourly generation/rejection investigation and independent system usability fixes.
> Current check: previous Luna directory batch32 done/28 accepted; one wrapped output,
> one stringified-boolean failure, two semantic disagreements. Batch schema unknown
> allowed the two format errors to finish. New directory-v2 adapter declares each
> filename stem and its question-derived type, with fresh IDs and preserved source
> groups/golds. Added quoted record field names to native parser/formatter/compiler
> targets; duplicate checks and exact type boundaries remain. Eight native typed
> references passed with complete visible inputs; no model calls. Default canonical
> static v2 publication is retained; typed draft is not a new published corpus.
> New64-case fresh directory batch: two Luna workers39580/39581, cap2/noGPU, runtime
> runs/generation-check-20260930-hourly/runtime-v3; state/queues/per-case exports at
> runs/luna-directory-typed-20260930/state.json. Fresh replies on both workers.
> Bonsai35895 remains cap4 with active decode. Boundary monitor39325 waits to switch
> collector to hourly runtime-v2 (JSON content oracle) without changing GPU server.
> Check rollout-state/log before changing supervisor; v2 excludes later quoted-name
> parser fix, which is present in Luna v3. Disk34GiB;1GiB floor on new workers.
> Hourly audit record runs/generation-check-20260930-hourly/check.json; next due about
> 2026-09-30 00:50 UTC (02:50 Berlin). Resume active sleep/check loop if interrupted.
> No student training; existing admission/source/split/student-token gates retained.


> **MINICPM ANALYSIS / JSON FILE FIX — 2026-09-30.** See
> plans/MINICPM_EVALUATION_ANALYSIS.md and saved failure-analysis.json.22/24 attempted,
> 11 final jobs,11 partial timeouts,2 unattempted; active reasoning/error loops dominate.
> Default JSON result-file oracle now compares strict parsed content; formatting/key
> order/newline differences pass, actual fields/types/strings/array order remain strict.
> Exact byte oracle remains opt-in, other file types remain exact. Provenance/reuse
> policy json-content/1 added; no broad data-quality bump or old positive data loss.
> Saved web release rescores to accepted;5/24 under content policy vs4/24 original.
> Original evaluation data/report retained, no GPU calls. New immutable runtime:
> runs/file-content-oracle-20260930/runtime-v1 for next campaigns; active v39/v4 stay
> frozen. No student training; held-out evaluation rows must never flow into train.


> **CURRENT — 2026-09-30, supersedes historical checks below.** Workflow visible-input
> fix is published as v2 via `data/teacher/workflowevals/static.manifest.json`.
> All4,805 cases independently prove complete source inputs remain visible at the
> final scripted answer;28,975 native approved decisions, zero held/unlinked.
> Source IR/gold/groups unchanged; synthetic reasoning masked. Old unsupported
> references now fail closed. Old files retained and replacements recorded.
> Frozen replay runtime `runs/workflowevals-visible-20260930/runtime-v4`;32k replay
> window avoids automatic compaction dropping early evidence. Final student token,
> split and rendered-pair dedup audits remain required. Inventory inclusion guard
> is clear in recipe-v3; this is not final training readiness. Four newer bundles
> now32,993 decisions/7,910 reducers (~23.97% before other layers/dedup); previous
> 63.43% claim is obsolete. Keep expanding real reducers toward25% unique decisions.
>
> DPO:43 causal native /2 pairs (35 failed_action,8 wrong_result) from116 selected
> candidates; `runs/dpo-audit-20260930/inline-handoffs-final.preference-pairs.jsonl`
> plus hash-checked manifest/audit. Source groups preserved. Labels not final until
> student rendering/token/split/dedup checks. Existing five /1 pairs remain migration
> backlog. Migration/source review/real candidate failure are separate dispositions;
> exclusion alone is never automatically a DPO negative. Recipe now includes offline
> correction and preference stages with audit manifests. No training started.
>
> MiniCPM evaluation completed its30-minute reservation; Bonsai restored PID35895,
> runtime-v39/queue-v31/cap4, fresh replies. Evaluation aggregate was overwritten by
> single-case exports: repaired accounting from11 durable completed jobs with exact
> IR checks,4 accepted/24 planned;13 missing include timeouts/not reached. Application
> probes not reached. Correct report `runs/minicpm-eval-20260929/report.recovered.json`;
> original erroneous zero-collected report retained. Recovery utility
> `scripts/recover_native_evaluation.py`; local evaluator now uses per-case exports.
> Do not describe missing/timeout cases as observed wrong answers.
>
> Original Luna pilot finished. New32-case all-directory batch with two provider
> workers PID36471/36472, cap2/noGPU, frozen visible runtime-v4. State/queues/logs:
> `runs/luna-directory-20260930/state.json`; disjoint fresh source cases, shortest
> first with three source families interleaved; per-case exports preserve records.
> Existing backoff/cooldown and1GiB disk floor retained. Confirm actual process and
> journal freshness before claiming continued activity.
>
> A Luna subagent is investigating a conservative10-case offline migration pilot
> for accepted current-IR teachers held only by obsolete runtime history. Originals
> preserved; candidate replay output is not automatically approved teacher data.
> Initial pilot had2 same-answer traces with changed observations and8 ambiguous
> replay ownership failures. Strict-v3 rejected all10:0 safe replay candidates.
> All history_migration rows are held by current admission, even direct inputs.
> Inspect `runs/native-history-migration-20260930` and strict-v3 follow-up.
> Latest recipe-v3 snapshot:2,758 trajectories/2,427 programs from5,520 completed
> files; zero missing carry-forward inputs/included quality blockers. No training.


> **DATA CARRY-FORWARD — 2026-09-30.** Both builders now automatically snapshot
> completed native teacher jobs, alongside older2,192 static reference trajectories
> and the four source bundles. New persistent artifact catalog and committed source
> policy expose migration/review/rerender backlogs, missing files and exclusions.
> Default recipes check that previously included artifacts are carried forward or
> have recorded replacement/disposition; explicit overrides are reported. Latest
> snapshot2,745 candidate trajectories/2,414 programs from5,490 completed files,
> not a final ready count. Catalog6,170 artifact paths (including raw JSON/archives), not distinct datasets.
> Reviewed planning recipe runs/data-lineage-20260930/recipe.json; no training.
> New pre-joint-training guard blocks the unresolved Workflow primitive evidence
> issue until fixed/held. See plans/DATA_LINEAGE.md and training/data_sources.json.
> data/teacher/data-inventory/current.json is the persistent view; immutable reports
> and per-file generated-snapshot ledgers preserve decisions and source hashes.


> **LATEST CHECK — 2026-09-29 22:17 UTC.** MiniCPM Q4_K_XL download is
> complete and SHA-256 verified; CPU tool-call smoke passed (`eval`, `2+3`).
> GPU evaluation has NOT started. Evaluator PID26597 is alive and waiting for a
> safe completed case boundary from Bonsai PID8981, runtime-v39/queue-v31.
> Current batch continues producing fresh replies; do not replace the supervisor
> while this evaluator watches its identity. Evaluation plan:24 heldout native
> cases plus13 application probes, then restore Bonsai. State/report directory:
> runs/minicpm-eval-20260929/. Older ENOSPC traceback in worker.log is historical.
> Luna PIDs22544/22545 are historical: current workers are26595/26596, resumed
> on queue-resume-v2.jsonl after disk-full failures. Only the two storage-failed
> case keys received reviewed retry suffixes; completed keys remain unchanged.
> Superseded V5/V6 audit files were losslessly archived with digest verification;
> ledger runs/luna-generation-20260929-two/storage-archive.jsonl. Both Luna workers
> are alive, one provider request each. Current filesystem has42GiB available.
>
> **OPEN QUALITY FINDING:** Workflow primitive scripted references can answer
> immediately although the initial rendered `state` is truncated. Directory
> full-visible-input proofs do not cover these4,194 primitive references. Their
> reported native approval counts must NOT be treated as a finished quality audit.
> Next: ensure complete state is visible before primitive scripted conclusions,
> independently prove that coverage, or hold affected static supervision. Preserve
> the source IR and live model generation; use a new frozen runtime for rebuilding.
> Canonical four-bundle count before this additional audit is12,471 approved native
> decisions/7,910 reducer decisions (63.43%); previous12,486/7,925 figures read an
> obsolete loose directory turns file instead of manifest-selected v9.


> **TWO LUNA GENERATION WORKERS — 2026-09-29 about21:57 UTC.** User explicitly
> increased Luna generation to two workers, superseding the earlier one-worker
> limit. PIDs22544/22545, gpt-6-luna/openai-codex, one request per process (two
> total), execution plans/low effort, frozen workflow runtime-v2. Disjoint queues:
> runs/luna-generation-20260929-two/worker-{1,2}/;32 cases each, eight reducers
> each (25% of64 roots). These collect actual model trajectories from the prepared
> WorkflowEvals pilot; static references are already integrated separately.
> First worker has two admitted completions; second has fresh model replies.
> Existing15s transport/rate-limit exponential backoff and provider-failure cooldown
> retained. New supervisor --min-free-mib1024 stops BEFORE a case if disk is low;
> restart same queue/journal after freeing space, no attempted-case skip added.
> State/PIDs: runs/luna-generation-20260929-two/state.json. Inspect actual processes
> and journals, not only startup status. Bonsai/evaluator unchanged; no GPU use.


> **WORKFLOWEVALS IMPORT — 2026-09-29.** User released the four retired eval
> repositories into training and confirmed Apache-2.0 for all (three licenses
> recorded as user-confirmed). Pinned acquisition complete:705 scenarios/13,105
> questions. Strict two-provider >=0.95 agreement/distribution/consensus gate:
> 4,194 typed judgments plus611 same-scenario directory batches;8,911 held source
> questions with reasons, original labels preserved. All4,805 native replays pass;
> 8,453 approved decisions/zero unlinked. Bundle data/teacher/workflowevals is
> auto-discovered by both training builders; default rematerialization matches.
> Typed source answers now train under a narrowly scoped verified-replay exception,
> synthetic reasoning masked. Original test provenance retained with exact pinned
> release allowlist; never use these training sources for evaluation again.
> First directory draft discarded read results and was withdrawn; current references
> show all files/pages and independently check complete visible evidence.
> See plans/WORKFLOWEVALS_IMPORT.md, course log and runs/workflowevals-20260929/.
> Final student token/rendered-pair/concentration audit still required; no training.
> BonsaiPID8981/v39/queue-v31 and MiniCPM evaluator were unchanged. Check evaluator
> state BEFORE changing supervisor. New teacher pilot is prepared, not activated.
> Older MuSiQue/QASPER/SciFact discarded-read references need later visible-evidence
> rebuilding; their scripted conclusions remain held. Do not globally enable direct
> answers for them. No new Luna worker. Acquisition uses bounded sequential batches.


> **AUTHENTICATED DOWNLOAD RECOVERY — 2026-09-29 about20:52 UTC.** User
> completed local HF login; authenticated whoami succeeds inside downloader.
> Token mounted read-only via HF_TOKEN_PATH, never printed or embedded in args.
> Earlier Xet transfer failed with CAS response-decoding error; both old watchers
> exited safely and Bonsai never paused. Replaced it with authenticated resolver
> plus validated resumable1MiB HTTP ranges/eight connections; completed ranges
> fsynced/journaled, final exact size/SHA required. Retained64.4MB from old valid
> ranges; latest71.8MB/1.596GB (4.5%),~0.20MB/s,zero new retries. No demonstrated
> token-driven speed gain yet; prior failure was not an HTTP429 quota error.
> Actual progress:`runs/sharp-minicpm5-discovery/download-state.json`;
> downloader container natlang-minicpm-download (1GiB/two CPUs), persistent
> range journal ranges.jsonl; script download-authenticated.py. Ignore sparse
> .part apparent size as a progress measure. Completion watcherPID16040;
> evaluation watcherPID16053 waiting for verified acquisition. Both resumed;
> BonsaiPID8981 continues unchanged. Earlier12763/14970 watcher PIDs historical.
> Evaluation state and eventual report remain runs/minicpm-eval-20260929/.

> **STUDENT EVALUATION QUEUED — 2026-09-29 about20:30 UTC.** User explicitly
> requested evaluation. Download remains incomplete; no model performance score
> yet. Durable evaluator PID14970 waits on acquisition/CPU smoke, then verifies
> SHA again, pauses exact BonsaiPID8981 only at a journaled case boundary,
> switches GPU to isolated MiniCPM on8082, and restores Bonsai server/supervisor
> with SAME v39/queue-v31/journal afterward. Thirty-minute GPU reservation;
> failures/timeouts/missing cases remain visible. Check state BEFORE touching any
> worker: `runs/minicpm-eval-20260929/state.json`, worker.log, eventual report.json.
> Pilot24 held-out cases across12 families, plus13 application probes. Same
> frozen v39 runtime, root seed42/temperature0; original gold preserved and24
> reference replays pass. Actor route pair was excluded for faulty reference.
> Honest-blocker probes now require an explicit relevant model blocker, not any
> exception. This is a greedy single-seed pilot, no Spark comparison yet.
> Acquisition completion workerPID12763 remains active; no student training.
> Shutdown: stop the acquisition completion worker and evaluator first, wait for
> evaluator cleanup/restoration, then stop the newly recorded Bonsai supervisor
> and servers. Never reuse historical8981 after evaluation restoration.

> **NEW STUDENT CANDIDATE — 2026-09-29 about20:15 UTC.** User requested
> peculiar-ragdoll's small Sharp MiniCPM and4-bit download/format/deployment checks.
> Matching release is Sharp-MiniCPM5-2B-GGUF (2.5B dense Llama, not sparse).
> Q4_K_XL1.596GB revision040713a6c4da5e58e7512e8e483d315caef41b73 is downloading
> in `natlang-minicpm-download`; fixed Xet concurrency8,2GiB RAM/two CPUs,
> slow network~0.3MB/s. Download is NOT yet verified and performance NOT assessed.
> Durable completion worker `runs/sharp-minicpm5-discovery/finish-acquisition.py`
> waits, verifies exact SHA/size, runs isolated CPU forced-tool-call smoke on8082,
> and stops its CPU server. Actual status:`runs/sharp-minicpm5-discovery/completion.json`.
> Read `plans/SHARP_MINICPM5_STUDENT.md` before testing/training. Template fixes:
> content-block query stale-reasoning reset and bare CDATA terminator escaping;
> coherent terse=false/thinking=true defaults; training terminator `<|im_end|>`.
> Renderer now closes real targets without a synthetic user query that erased
> target reasoning; renderer version2/cache identity changed, published static
> payloads unchanged.12 focused renderer checks pass; actual tokenizer renders
> one BOS/correct EOS and retains reasoning. Bonsai generation remains on GPU;
> `scripts/serve_minicpm.sh --gpu` refuses while Bonsai is running. No training.

> **CURRENT CHECK — 2026-09-29 19:40 UTC.** Bonsai PID8981 now runs frozen
> v39/queue-v31, cap4; GPU100%/7,457MiB. Compiler-poisoned case52 freshly
> succeeded (43 turns); the original failure remains. Current batch is progressing
> (56 saved turns/21 fresh replies);353 pending entries/742 roots including current.
> Luna v38/queue-v27 finished its five retries and exited; all five rejected, no
> new worker started. Fresh audit includes separate repair jobs previously missed:
>555 results/365 admitted/1,703 approved decisions/191 held/zero unlinked.
> Artifact: `runs/check-20260929-190056/generation-quality-review.json`.
> MuSiQue57 and783 have source-question/gold disputes (unveiling vs launch;
> US vs English Portsmouth); hold adjudication is still outstanding, including
> propagation into static/queued data.801 is an exact-wording false negative.
> Correction to the earlier span-F1 claim: partial overlap remains excluded by
> the training gate and is insufficient proof of equivalence. Future adapter is
> restored to normalized exact plus original source aliases. Published V9 unchanged.
> Next: propagate verified source holds, curate answer equivalence independently,
> and include repair output paths explicitly in the next SFT build. No training.

> **GENERATION CHECK — 2026-09-29 about19:00 UTC.** BonsaiPID5386 on
> frozen v37/queue-v29 continues with cap4; GPU100%/7,457MiB. One long case
> is progressing; two post-reboot hard timeouts reached their wall budgets with
> increasing decode/checkpoints, not silent stalls. Luna's resumed v37 queue-v26
> finished all98 entries cleanly and stopped. Since the previous audit,125 new
> teacher results:115 admitted,559 approved decisions,zero unlinked.
> A real scope-compiler bug let final comma expressions pass their second value
> as the eval bindings argument, poisoning later evals with numeric names. The
> parenthesization fix is frozen in v38; a defensive invalid-name guard is frozen
> in v39. Exactly one new boundary monitorPID8804 waits for Bonsai's current case,
> then switches to v39/queue-v31; actual state:`runs/check-20260929-190056/v39.rollout-state.json`.
> Queue-v31 starts with one reviewed fresh retry of the poisoned case; original
> failure remains. One Luna workerPID8658 on v38/queue-v27 is trying5 reviewed
> MuSiQue misses, separate artifacts, cap1. For future source builds, MuSiQue uses
> a strict0.9 QA token-overlap check so a correct long answer with a named subject
> is not rejected for formatting alone. Existing result labels and static V9 remain.
> See `runs/check-20260929-190056/` and course log for details. No training.


> **GENERATION RESUMED — 2026-09-29 16:51 UTC /18:51 Berlin.**
> User requested restart after reboot; this supersedes the stopped state below.
> Bonsai server healthy:4 slots,6GiB container cap,1,536MiB host cache, GPU100%/7,455MiB.
> Saved runtime/checkpoint/V9 payload checksums verified. Bonsai recoveryPID5100 uses
> v34 and the one-entry interrupted-batch queue; existing indices53/54 are retained,
> only52/55 remain pending. New boundary monitorPID5101 switches to v37/queue-v29
> after recovery; actual new PID/state: `/home/werg/natlang/runs/restart-20260929-165057/v37.rollout-state.json`.
> LunaPID5040 on v37/queue-v26/cap1 resumed537 and completed it accepted in20.3s;
> next case540 started. Exactly one supervisor per teacher; existing journals retained.
> Restart metadata/logs: `/home/werg/natlang/runs/restart-20260929-165057/restart.json`; pointer:`runs/RESTART_LATEST`.
> Old shutdown/rollout PIDs remain historical. Keep QASPER generation pause/source
> holds; continue quality monitoring and25% reducer target. No training.


> **USER-REQUESTED SHUTDOWN — 2026-09-29 11:28 UTC /13:28 Berlin.**
> Generation workers, boundary monitor and `natlang-bonsai` container are STOPPED.
> Nothing should automatically restart. The old PIDs below are historical; never signal
> them or reuse the old rollout config after reboot. Shutdown completed in under3 minutes.
> Authoritative checkpoint: `/home/werg/natlang/runs/shutdown-20260929-112702/checkpoint.json`; pointer:`runs/SHUTDOWN_LATEST`.
> Read **Restart after shutdown** below before resuming. Current published data is V9;
> quality changes committed as `644a96f`. Unrelated adaptation/package development edits
> remain in the working tree and must not be reverted or accidentally committed.


> **Quality sweep / paused QASPER generation, 2026-09-29:** Luna subagent audited170
> SciFact cases; corrected multi-document omissions independently. Root holds15 further
> cases pending adjudication, preserving gold (see course log/root-adjudication.json).
> Also hold SciFact657/748 and FOLIO406. QASPER live generation paused for an
> extractive-equivalence oracle; static read decisions remain. Prompt now preserves
> final newlines in span edits and explains initial iteration stop/pagination at all
> NL depths. Frozen v36 failed depth-prompt test and was never deployed;v37 passes21
> focused Node checks. Latest teacher snapshot421/246 admitted/1,070 approved/zero unlinked.
> V9 published:1,306 expansion cases/3,536 approved decisions;8,531 combined static
> decisions,42.80% reducers; zero duplicates/holdout overlap. V8 preserved. LunaPID2113145 runtime-v37/queue-v26/cap1; BonsaiPID1130311 runtime-v34
> remains decoding its batch; v37/queue-v29/cap4 rolls only at boundary. Verify actual
> `runs/generation-check-20260929-1055/v37.rollout-state.json`; monitorPID2112978.
> Audit artifacts/corrected Luna report/course decisions logged in
> `plans/DIRECTORY_EXPANSION_20260929.md`. No training.


> **Rejection audit / quality tightening, 2026-09-29:** both workers progressing;
> 53 Luna completions with no operational failures since v33, two Bonsai batches
> completed plus one active-decode hard timeout. Hold SciFact claims466/576 and
> FOLIO story24 without relabeling. TreeDST currently lacks an independent branch
> ontology: retain existing literal-slot replacements; hold additions/deletions and
> unspecified structural transitions, including historical positives/static cases.
> This deliberately retains21/300 TreeDST static cases. V7 published and verified:
> 1,323 expansion cases/3,553 approved decisions; combined8,548 unique static decisions,
> 42.91% directory reducers,42 tree-edit decisions; zero duplicates/holdout overlap.
> V6 preserved in runs/generation-check-20260929-0926/published-v6-preserved/.
> Exact structured-text prompt guidance fixes one fresh CommitPack repair (accepted,
> four actions), original failure preserved.52 frozen Node/25 Python checks pass.
> Luna v34/queue-v24 PID1100034; Bonsai v31 PID701072 awaits v34/queue-v27 at
> its progressing batch boundary. Verify actual state before action:
> runs/generation-check-20260929-0926/{v34,payload-repair}.rollout-state.json.
> Latest teacher audit240 results/128 admitted/575 approved decisions/zero unlinked.
> Further detail: plans/DIRECTORY_EXPANSION_20260929.md. No training.


> **Luna exponential backoff, 2026-09-29:** request transport retries now use 15s base,
> 30s cap; rate-limit retries use 45s base,120s cap. Both double with jitter clamped
> after jitter. Supplied Retry-After/date or SDK error delay is a minimum, even above
> caps, and survives exhaustion/timeouts/restarts through durable retry/error deadlines.
> Retry waits are abortable and logged; watchdog recognizes intentional waits while
> hard case budgets remain. Failed provider runs get30/60/120/240/300s nominal cooldown
> with jitter, persisted in supervisor journal; clean completion resets it, including
> ordinary model rejection. One Luna worker/request remains. Frozen v33 derives from
> immutable v31 plus collector/retry files only, excluding concurrent adaptation work.
> Boundary rollout complete: Luna PID731206 on v33/queue-v22 (verify before action);
> runs/directory-expansion-20260929/v33.rollout-state.json. First v33 case completed.
> 21 frozen Node /13 Python checks and build pass. No training.


> **Post-retry quality hold:** TAT-QA UUID f944b361-6e00-45c8-a7e1-1f5c6e0fd6b1
> admits signed-average versus expense-magnitude interpretations. Hold source unchanged.
> V6 refresh removed its six static decisions:4,113 added decisions /9,108 total
> audited static decisions,40.29% directory reducers, zero duplicates/holdout overlap.
> Corrected bundle published; V5's9,114 preview remains historical.
> Runtime-v31 rolls at boundaries to Bonsai queue-v26/cap4 and Luna queue-v22/cap1;
> actual state/PIDs: runs/directory-expansion-20260929/v31.rollout-state.json.
> V6 publication and frozen v31 validation pass. Both workers migrated after completing their
> progressing old batches. Bonsai PID701072 / Luna PID695902 (verify state before action). Supervisor stall detection now recognizes
> actual GPU decode progress before a reply checkpoint exists; hard budgets remain.
> Latest22 Python and29 source-review/oracle checks pass. No training.


> **Source expansion / four Bonsai requests, 2026-09-29:** user target25% directory/file
> reducer decisions in the final admitted train mix; tree edits count separately. Expanded
> CommitPack/TAT-QA/MuSiQue plus QASPER/SciFact/TreeDST supply1,605 new cases; static
> native replay gives4,119 approved decisions and1,049 held. V5 token audit passes all;
> combined static preview9,114 decisions,40.33% directory reducers, zero duplicates/
> holdout overlap. Corrected expansion published in data/teacher/directory-expansion/. Builders now stream to bound
> RAM. Fixed TreeDST sibling-order scoring; JSON whitespace scoring correction and
> ambiguous QASPER-negative holds now deployed in runtime-v30 at boundaries. Bonsai actively
> uses4slots/global cap4, 1,536MiB host cache/6GiB cap, four roots/collector. Initial
> sample: mean3.68 active slots,18.27tokens/wall sec, GPU7,457MiB (~7.28GiB)/initial server RAM2.85GiB.
> Keep four pending longer evidence, one Luna/cap1. State/PIDs:
> runs/directory-expansion-20260929/{v30.rollout-state.json,json-repair.rollout-state.json}.
> Bonsai queue-v25/cap4; Luna queue-v21/cap1 includes two reviewed JSON-format retries.
> Final66 Node/19 Python checks and builds/browser types pass.
> Course decisions/remaining work: plans/DIRECTORY_EXPANSION_20260929.md. No training.


> **Rejection follow-up 08:20,2026-09-29:** new failures have no scope/runtime errors.
> Luna repeatedly reinterpreted a country as a movie despite role guidance. Added general
> empty-search/helper guidance: reread original question, roles and documented store scope;
> rephrasing the same unsupported premise is not new evidence. No answer hints added.
> Hold FOLIO422: source FOL drops customer guards; James is not established as a customer.
> Runtimev27 rolls at case boundaries on unchanged queuesv20/v16; actual status/PIDs:
> runs/bonsai-recovery/runtime-v27.rollout-state.json.161 Node checks/builds/types pass.
> Steering audit path runs/rejection-steering-20260929-0918/ (directory label is historical,
> check actually08:20 Berlin). Keep hourly loop; next scheduled check08:43, no training.

> **Hourly check 07:36, 2026-09-29:** both workers progressing. Fifteen accepted
> runs yield207 linked decisions, no unlinked exports; Luna fixed shared-inbox deduplication.
> Hold FOLIO417 (inclusive/exclusive-or) and378 (negation scope), plus four ambiguous
> BANKING77 fee/amount/reversal messages. Model errors retained separately; no relabeling.
> V26 queues: Bonsai queue-v20 / Luna queue-v16,176/13 pending at preparation.
> Twelve unfinished groups transferred to the single Luna queue, bounded to at most two
> cross-teacher attempts per source group. Integrated static bundles have zero new holds.
> Fixed boundary-monitor failure when an old queue naturally exhausts: reusable
> scripts/roll_teacher_runtime.py verifies all old keys finished before resuming; unexpected
> exits with unfinished entries fail closed. Luna resumed PID640685 on v26; actual state/PIDs:
> runs/bonsai-recovery/runtime-v26.rollout-state.json. Audits: runs/hourly-check-20260929-0736/.
> Builds/browser types,161 Node and9 Python checks pass. No training; keep hourly loop running.

> **Hourly check 06:30, 2026-09-29:** both workers progressing, Bonsai warm RAM1.97GiB.
> Nine accepted results this hour yield168 linked decisions; three rejected results and
> four incomplete/timeouts audited. Hold FOLIO409 after English countermodels and source
> negation mismatch; hold EntailmentBank LEAP__7_10338 for reversed implication proof;
> hold two more ambiguous pending-dollar BANKING77 messages. Source holds filter73 Bonsai
> and7 Luna queue entries (including completed entries); preserve originals and gold.
> Eval timeout guidance now states queued child time counts and started work is not cancelled.
> Large reducers spent budgets on repeated/poorly specified judgments, not a frozen provider.
> Twelve eligible unfinished groups added as fresh Luna roots, without extra workers.
> V25 boundary rollout: Bonsai queue-v19 / Luna queue-v15, state/PIDs at
> runs/bonsai-recovery/runtime-v25.rollout-state.json. Audits: runs/hourly-check-20260929-0630/.
> Builds/browser types and161 Node checks pass; current static bundles have zero new holds.
> No training. Continue hourly checks until stopped.

> **Hourly check 05:06, 2026-09-29:** navigation retry reached its goal, but its
> intermediate wrong-object action is held at episode level; valid source remains eligible.
> Fresh supplier retry exports cleanly; 24 accepted runs retained 488 decisions with zero
> unlinked actions. Found/fixed native-export bypass of strict partial-correctness policy:
> 61 historical accepted result files with partial answers/files/review flags now export
> zero decisions in all modes. Integrated static bundles have no new policy holds.
> Added source holds for FOLIO423's missing student guards and ambiguous BANKING77 pound charge.
> Repeated action shapes across child calls do not imply a stall; queue logs now also count
> full request hashes. Twelve fresh source groups extend the single Luna queue.
> V24 boundary rollout: Bonsai queue-v18 / Luna queue-v14, state/PIDs at
> runs/bonsai-recovery/runtime-v24.rollout-state.json; audit runs/hourly-check-20260929-0506/.
> Builds/browser types,160 Node and6 Python checks pass. Rebuild earlier generated training
> artifacts through current gates before use; no training. Continue hourly checks until stopped.

> **Hourly check 03:50, 2026-09-29:** both workers healthy; proof chain19 repair
> admitted. Hold FOLIO454 and three newly ambiguous BANKING77 messages after source audit.
> TextWorld v2 shows available exits and asks for exploration notes; world cases use
> 48 turns /384+ requests /1800 sec. Nine pending source groups added to the single Luna queue.
> Fixed native decision/action joins with explicit invocation metadata; old ambiguous rows
> excluded and one fresh retry queued. Identical-opening preference replays remain held.
> Latest runtime-v21, Bonsai queue-v17 / Luna queue-v13; state/PIDs:
> runs/bonsai-recovery/runtime-v21.rollout-state.json. Audits: runs/hourly-check-20260929-0350/.
> Builds/browser types,158 Node tests and5 Python queue tests pass. No training.
> Continue the hourly sleep/check loop until stopped.


> **Hourly check 02:33, 2026-09-29:** repaired dropped `__proto__` data keys,
> qualified service alias annotations, and stale mutable inline captures across evals.
> Old affected runtime failures excluded from training negatives; fresh reviewed retries queued.
> Hold ambiguous KQA Pro train:33143 and FOLIO477 without relabeling.
> V19 boundary rollout: Bonsai queue-v15 / Luna queue-v11, actual state/PIDs at
> runs/bonsai-recovery/runtime-v19.rollout-state.json. V17 was superseded before deployment.
> Fixed the collector slot/journal checkpoint race; expanded 156 Node tests and builds/types
> pass. Fresh mutable-budget retry accepted. Audits: runs/hourly-check-20260929-0233/.
> One Luna worker, no training. Continue the hourly sleep/check loop until stopped.


> **Hourly check 01:25, 2026-09-29:** v15 movie-awards retry and board retry
> are training-admitted; country query remains an incomplete model interpretation
> failure (actual input verified visible). Hold FOLIO 416 after countermodel review.
> New queue rotation spreads valid cases across families and source groups without
> dropping cases or changing budgets. V16 boundary rollout uses Bonsai queue-v13 /
> Luna queue-v9. Actual state/PIDs: runs/bonsai-recovery/runtime-v16.rollout-state.json.
> Audits/order ledgers: runs/hourly-check-20260929-0125/. One Luna worker, no training;
> continue the hourly sleep/check loop until the user stops it.

> **Hourly monitoring, 2026-09-29:** both workers active on v14; smaller Bonsai
> server runs two slots, warm RAM about 2.5 GiB and GPU memory 598 MiB lower.
> All four numeric CommaQA retries accepted. V15 prepared: movie specialist
> ownership now derives from each world's stores; canonical relation facts,
> v3 movie-schema hold, FOLIO 348/377 holds, and reset irreversible repeated-write
> board prefixes. New queues Bonsai v12 / Luna v8. Actual rollout state/PIDs:
> runs/bonsai-recovery/runtime-v15.rollout-state.json. Checkpoint and audits:
> runs/hourly-check-20260929-0013/. Hourly check/sleep loop continues; no training.

> **Bonsai RAM reduction queued (2026-09-28):** six server slots exceed the
> current two-request collector. Launcher now defaults to two slots / 1.5 GiB
> host prompt cache / 6 GiB memory ceiling, retaining 53k shared GPU KV.
> Safe server restart monitor and actual status: runs/bonsai-memory-20260928/
> state.json and restart.jsonl. It resumes the same v14 queue and journals;
> do not treat the historical v14 rollout PID as current after this restart.
> Steady-state savings still need measurement after cache warmup.

> **Latest check-in: v14 prepared (2026-09-28):** corrected CommaQA numeric
> upstream sport inversion (all 64,000 nationality renderings affected), added
> literal retrieval/schema guidance and completed-effect receipts. Numeric
> contracts below v3 are held until migrated; raw source and gold preserved.
> Added pending FOLIO 56/8 and αNLI Ciana reviews. Read the final section of
> REJECTION_FOLLOWUP_20260928.md and runs/rejection-checkin-20260928-late/.
> Bonsai queue-v11 / Luna queue-v7 use distinct v14 jobs with reviewed retries.
> Boundary rollout actual state: runtime-v14.rollout.jsonl / rollout-state.json.
> One Luna worker/request, Bonsai independent, no training.

> **Rejection follow-up (2026-09-28 evening):** see
> [REJECTION_FOLLOWUP_20260928.md](REJECTION_FOLLOWUP_20260928.md).
> Fixed two false admission rejections, persistent-state error guidance, mail
> declaration/idempotency validation, supplied highlighter iterators, CommaQA
> specialist contracts and computed references, and per-case world turn budgets.
> Six αNLI source inputs are held for adjudication; old numeric contracts are held
> until migrated. Earlier ready artifacts must be rebuilt under these new gates.
> Prepared Bonsai queue-v8 / Luna queue-v4 and v11 runtime rollout at completed
> case boundaries. Actual PIDs/status: runs/bonsai-recovery/runtime-v11.rollout-state.json
> and runtime-v11.rollout.jsonl. Raw v10 jobs remain immutable; one Luna worker only.
> Later arrivals exposed English/formalization mismatches in FOLIO stories 337 and
> 162: both are now held, without changing gold. Latest rollout is **v12**, Bonsai
> queue-v9 / Luna queue-v5; actual boundaries/PIDs are in runtime-v12.rollout.jsonl
> and runtime-v12.rollout-state.json. Luna's initial v11 highlighter retry is allowed
> to finish. Verification now includes 110 focused Node tests.
> Final export/oracle follow-up: native materialization honors source holds directly;
> legacy highlighter gold is held (four seeds excluded by source freezer v3).
> All 2,192 static references still replay/admit; 127 focused Node checks pass.
> Latest rollout: **runtime-v13**, Bonsai queue-v10 / Luna queue-v6; read
> runtime-v13.rollout.jsonl and runtime-v13.rollout-state.json for actual PIDs.
> Stop the already source-held, repetitive handoff 23 without classifying it as a
> training negative. Other progressing cases migrate at boundaries. First world
> retry succeeds with score 100 / 22 moves. No training started.

> **Modern adapters / strict static quality audit (2026-09-28):** read
> [MODERN_STATIC_ADAPTERS.md](MODERN_STATIC_ADAPTERS.md). Modern decision, state, scene,
> numeric, synthetic/fold and leaf adapters are implemented. Only the verified 72-case
> pilot / 325 decisions is automatically connected, alongside the existing 51-case
> directory pilot. FinQA, sales/Typed labels, semantic synthetic programs and all 620
> joined leaf references remain held. Raw annotations/splits/licenses are preserved.
> Rebuilt directory IR matches exactly; all 2,192 old static references replayed.
> Sharp audit discovered 661 duplicate reference pairs: final audit now removes them
> and holds train/holdout-connected rows (4,513 unique reference decisions; combined
> static audit: 4,995 unique decisions, max 6,356 tokens). Ready
> artifacts and rejection ledgers are under runs/modern-adapters-20260928/ and
> runs/integrated-quality-audit-20260928/. No training/model calls started by this work;
> existing Bonsai and single Luna supervisors remain on frozen v10.

> **Static data recovery audit (2026-09-28):** see `plans/STATIC_DATA_RECOVERY.md`.
> Reconnected 2,192 static reference trajectories to the staged recipe; 5,174 decisions
> currently pass native admission, before dedup/split/token gates. Held six obsolete
> default code turn snapshots (120 rows), added fresh saved-source replay, recovered
> 18 current turns, and fixed source `arguments` fidelity / missing failure ledgers.
> Large decision/state/scene corpora and the 620-entry leaf bank survive but need
> modern adapters. Evidence: `runs/legacy-data-audit-20260928/`.

> **Static source pilot implemented (2026-09-28):** 51 admitted directory cases: WorkBench 8, CommitPackFT 12,
> TAT-QA 11, MuSiQue 12, SWE-smith 8 independent file-creation tasks. Zero generation model calls. The final
> Sharp audit admits all 157 rendered decisions at max-len 8192 (largest 6356); 32 direct scripted decisions
> stay held in native evidence. Bundle: data/teacher/source-backed/static.manifest.json; audited artifact:
> runs/source-backed-20260928/verified/sharp.ready.jsonl. Both build_lora_sft.sh and the staged recipe now
> discover/validate this bundle. No training or new teacher queue launched. Whole SWE traces imported: zero;
> eight imported source operations are explicit scoped tasks. Read DIRECTORY_TASK_SOURCES.md for commands,
> provenance, source holds, tokenizer binding and the remaining reasoning-trajectory work. Current collectors
> remain on frozen v10; changes here are in builders/admission/materialization, not a live-worker migration.

> **Source research (2026-09-28):** [DIRECTORY_TASK_SOURCES.md](DIRECTORY_TASK_SOURCES.md) records
> new directory-task sources and existing trajectories for offline IR conversion. Start with a successful-trace
> portability audit for Nebius OpenHands / SWE-smith; WorkBench, CommitPackFT and TAT-QA lead task expansion.
> Research samples under runs/source-research-20260928/ are not training data. The implemented pilot above supersedes this research checkpoint.
> Workspace-Bench and MuDABench remain evaluation-only; preserve source success separately from native replay
> validation. Latest live check: Bonsai handoff 790, Luna case 11, same v10 supervisors and single Luna worker.

> **Active runtime-v10 rollout (2026-09-28):** both workers now run the frozen d69af3a runtime, including
> criterion-preserving delegation guidance and pending-source collection/admission protections. All 455 frozen
> file hashes verified. Luna supervisor PID 332686 uses queue-v3.jsonl, existing v2.journal.jsonl, new pending-job
> paths ending .v10; 150 eligible entries / 146 remaining at migration. One worker / one request in flight.
> Bonsai supervisor PID 332210 uses queue-v7.jsonl, existing journal.jsonl, pending paths ending .v10;
> 558 eligible entries / 504 remaining at migration. Completed entries retain old evidence paths.
> Removed three affected Luna entries and one affected Bonsai entry into v3.held.json / v7.held.json for source
> review; these counts include previously attempted tasks. Original source shards, labels and raw evidence remain.
> Migration waited for existing case boundaries. Luna extraction case 0 hit its 900-second deadline (106 replies,
> 908 seconds including shutdown); Bonsai handoff 555 hit 600 seconds (18 fresh replies, 603.1 seconds including
> shutdown). Both remain unfinished checkpoints, excluded from training/negatives, needing a reviewed retry.
> Current cases started on v10: Luna 4, Bonsai handoff 552. Rollout audit/provenance/state files are under
> runs/bonsai-recovery/runtime-v10.*. Older runtime/checkpoint banners below are historical.

> **Latest payment review checkpoint (2026-09-28):** both supervisors remain on immutable runtime-v9.
> Luna: 5 completed attempts / 148 remaining; 2 training-admitted, 2 partial-agreement holdouts, 1 wrong total.
> The 142-message task finished in 1,060.5 seconds without a collector error; 13,386 versus 13,888 is one missed
> judgment, not a complete repair. Bonsai: 53 finished attempts / 506 remaining (46 collector completions,
> 7 historical hard timeouts). Three ambiguous BANKING77 inputs are now in the shared source-review registry;
> main generation, admission and handoff/pair builders hold affected tasks pending adjudication. Raw labels/results
> remain preserved. New parent-criterion preservation prompt is built and checked in the main checkout, **not yet
> deployed to the running v9 workers**. Migrate at a safe boundary; do not interrupt a large progressing case.
> Payment mismatch artifacts are in runs/rejection-audit-20260928/payment-followup/. Read the latest decision log.

> **Latest user authorization (2026-09-28): restart exactly one Luna repair worker.**
> This supersedes the earlier stop instruction. Use the current audited, deduplicated repair queue in
> `runs/luna-repair-20260928/queue-v2.jsonl`, with frozen runtime-v9, one collection worker and one model request in flight.
> Bonsai continues independently. Quality-pending, retired and unverified-contract cases stay excluded.
> Current Luna supervisor PID 286423: 153 queued retries, with the already admitted repair excluded, plus 3 legacy
> source cases needing reconstruction in deferred.jsonl. Nine other legacy tasks migrated successfully.
> Source shard is immutable while the collector runs; collector.log, jobs/ and summary.json record progress.
> Earlier Luna checkpoint: 3 finished / 1 admitted / 2 task failures, 2 request-budget errors. Raw jobs preserved.
> Both collectors now use runtime-v9: paired planning/action scheduling, ordered checkpoints and shared-variable
> guidance. Luna has sized request budgets and bounded case/inactivity deadlines. Fourteen payment tasks migrated
> to explicit card/direct-debit/cash-withdrawal scope; old wording is quarantined. Read the latest decision log.
> Bonsai supervisor PID 287222 uses queue-v6 (46 finished attempts / 513 pending at migration checkpoint).

> **Latest 2026-09-28 checkpoint:** use `runs/bonsai-recovery/queue-v5.jsonl` and frozen `runtime-v7`.
> It prioritizes the verified SMS repair handoff; 559 entries / 20 completed attempts / 539 pending at migration.
> Read DATA_QUALITY.md's fresh checkpoint. Token usage is now preserved in replay, fixing false handoff rejection
> from changed compaction timing. Folder guidance permits direct/delegated judgments and reuse of completed work.
> New output paths preserve interrupted extraction partials. Rewrite/answer-equivalence cases await independent
> review and are held out. Luna stays stopped; no training has started.

> **2026-09-28 data quality and slow-case review:** Bonsai now uses `queue-v4.jsonl` and frozen
> `runtime-v5`, with the existing append-only journal. New jobs have distinct paths; raw histories are preserved.
> Read [DATA_QUALITY.md](DATA_QUALITY.md) and the latest decision log before restarting or building data.
> The queue has 558 entries before journal skips: six distinct verified folder replacements, two measured FOLIO
> retries, and the filtered recovery work. Unsupported premise-removal and legacy CUAD probes are excluded.
> Inactivity (no saved reply for five minutes) is distinguished from hard deadlines; multi-item cases get larger
> bounded budgets. Quality-pending and partial benchmark results are held out of training/negative pairs.
> Correct direct/delegated/mixed results remain eligible. Luna stays stopped; no model training has started.

> **2026-09-28 full rejection audit:** Bonsai now uses `runs/bonsai-recovery/queue-v3.jsonl`
> and frozen `runs/bonsai-recovery/runtime-v4`, with the same append-only journal.
> Six reviewed probes precede the 546-entry recovery queue; completed attempts are skipped. Luna remains stopped.
> User's delegation policy: **three active ad hoc layers per root; every pre-existing file-backed `.nl`
> function starts a fresh root budget**. The third layer has a separate prompt/help variant and cannot create
> a fourth. Inline nl, Python nl and delegate share the limit. Correct direct and delegated answers can both
> be admitted. Execution policy provenance version 2 prevents silent reuse under the changed behavior.
> Full audit: `runs/rejection-audit-20260928/`; read `plans/REJECTION_REVIEW.md` and the latest decision log entry.

> **2026-09-28 rejection investigation:** Luna remains stopped. Current Bonsai queue is
> `runs/bonsai-recovery/queue-v2.jsonl` (546 eligible entries before journal skips), using
> `runs/bonsai-recovery/runtime-v3` and the existing `runs/bonsai-recovery/journal.jsonl`.
> Seven suspicious rejections replayed with correct answers: all belonged to retired `inline_type_repair`.
> Seven retired entries were removed from the queue; raw evidence remains excluded from training.
> Seeded handoffs now let a teacher replace the planted failure without reinjecting it. Seeded handoff
> provenance version 2 prevents reuse of incompatible old rows/partials. See the latest decision log entry.

> **2026-09-28 midday:** Luna is stopped at the user's request. Bonsai now runs the bounded recovery queue:
> `runs/bonsai-recovery/queue.jsonl` (553 unfinished cases), journal `runs/bonsai-recovery/journal.jsonl`.
> Runtime frozen at `runs/bonsai-recovery/runtime`; supervisor `scripts/run_bonsai_queue.py`.
> Limits: 600 seconds/case plus 10-second shutdown grace, 2 in-flight model requests, 128 new model requests/case.
> The old CUAD probe collector and launcher were stopped. Probe comparison remains incomplete and deferred;
> no file-tool default was changed. Read the latest decision log entry before restarting anything.

> **2026-09-28 morning:** user reduced Luna to **3 total workers** (folder 1, handoffs 1, redo 1).
> Active runner is now `/home/werg/natlang-generation-runner/ts-host`, with the streaming export fix.
> Bonsai train4 is complete; folder 102/106 saved, handoffs 282/833 saved, redo 12/13 saved at restart.
> Bonsai surface probe resumed at one worker after shared-context failures. See the latest decision log entry.

> Continuation 2026-09-27 night: see [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).
> Bonsai generation and **five total Luna workers** resumed (2 folder, 2 handoffs, 1 redo).
> This supersedes the two-worker limit below. Isolated runner refreshed; train4 restart now needs explicit reuse
> of `train4.resume.jsonl` (redo: `train4-luna.resume.jsonl`) with
> `--reuse-surfaces 58ec5990ab293f66d29b2a34115663a2371cf4b8fa5e10f3dc43ecdbb5b232d7`.
> Correction: current collectors publish after each job, but exports cover only their selected range. Use
> `node scripts/snapshot_teacher_jobs.mjs JOBS_DIR OUTPUT.jsonl` to gather all completed ranges.

Everything was shut down gracefully for a machine power-off: collectors finished their current step (partial runs are
journaled in each `*.jobs/` directory and resume on restart), the Bonsai server container (`natlang-bonsai`) was
stopped. Nothing is lost by restarting any collector below with the same arguments.

## Standing instructions (from the user)

- Kill processes by exact PID, never `pkill -f`/`pgrep -f`. Collectors: `ps -eo pid,args | awk '$2=="node" && /dist\/teacher\/cli.js/'`.
- Verify builds and tests before committing; commit freely. `while` stays forbidden in eval, bash and Python.
  Network is on by default in every tool. Browser parity is required; natlang runs in a worker in the browser.
- No junk data: delete backups, moved-aside dirs and superseded builds once their replacement is verified.
- Fix issues generally, not narrowly. Tool shapes are chosen by what our models do best (measured), nothing else.
- **One Luna repair worker is authorized by the latest user request.** Keep Luna at one collection worker and
  one model request in flight, including children. Bonsai continues the bounded recovery queue.
- Student: Sharp-Spark-X2.5-4B, trained at max-len 8192 (QLoRA, see `scripts/train_lora.py`, 8 GB GPU, 14 GB RAM).
  The user chose to keep generating before training.
- Another agent works in this checkout at times (it built the directory-reducer runtime). Check `git status` before
  committing; stage only your own hunks when files are shared.

## Machine layout

- Main checkout `~/natlang`. Runner checkouts, so collectors keep running while `ts-host/dist` is rebuilt:
  - `~/natlang-runner/ts-host`: rsync'd copy used by the Bonsai collector and the redo loop (older code: no handoff
    support). Refresh with `rsync -a --delete --exclude node_modules --exclude dist ts-host/ ~/natlang-runner/ts-host/`
    and `npm run build` there, only while no collector runs from it.
  - `~/natlang-folder-runner`: git worktree at `ddb136e` (current main), used for the folder batch and handoffs.
    Update with `git -C ~/natlang-folder-runner checkout --detach main && (cd ~/natlang-folder-runner/ts-host && npm run build)`
    while nothing runs from it. Remove it (`git worktree remove`) when no longer needed.
- Bonsai teacher server: `nohup scripts/serve_bonsai.sh > runs/bonsai-server.log 2>&1 &` (port 8081, 6 slots, 53k KV);
  wait for `curl -s 127.0.0.1:8081/health`. It uses ~7.7 GB of the 8 GB GPU; nothing else fits beside it.
- Memory is tight (14 GB; Bonsai's server takes ~6 GB RSS). Keep node scripts at `--max-old-space-size` 6000 or less.
- Times: `date`/`ps`/files are local (CEST); trace `observed_at` and some logs are UTC.

## Restart commands (all resume where they stopped)

```bash
cd ~/natlang && nohup scripts/serve_bonsai.sh > runs/bonsai-server.log 2>&1 &
# Bonsai: the 13 train4 cases still missing are all in 0-74
cd ~/natlang-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/inline-curriculum/train4.ir.jsonl \
  ~/natlang/runs/inline-curriculum/train4.jobs ~/natlang/runs/inline-curriculum/train4.results.jsonl \
  --model-id Ternary-Bonsai-2-27B --root-seed 909 --server http://127.0.0.1:8081 --start 0 --limit 75 \
  --max-turns 20 --workers 6 >> ~/natlang/runs/train4.log 2>&1 &
BONSAI=$(ps -eo pid,args | awk '$2=="node" && /train4\.jobs/ {print $1}')
# Luna redo of Bonsai's rejected cases (1 worker)
cd ~/natlang && NATLANG_TS_HOST=~/natlang-runner/ts-host nohup scripts/teacher_redo_loop.sh \
  runs/inline-curriculum/train4.ir.jsonl runs/inline-curriculum/train4.results.jsonl runs/inline-curriculum/train4-luna \
  --while-pid $BONSAI -- --provider openai-codex --model-id gpt-6-luna --root-seed 909 --workers 1 \
  --context-tokens 16384 --max-turns 20 --reasoning-effort low --execution-plans >> runs/train4-luna-redo.log 2>&1 &
# Luna folder batch (1 worker)
cd ~/natlang-folder-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/folder/gen1.ir.jsonl \
  ~/natlang/runs/folder/gen1.jobs ~/natlang/runs/folder/gen1.results.jsonl --provider openai-codex --model-id gpt-6-luna \
  --root-seed 1101 --all --workers 1 --context-tokens 16384 --max-turns 20 --reasoning-effort low --execution-plans \
  >> ~/natlang/runs/folder/gen1.log 2>&1 &
```

Paused (resume only within the 2-worker Luna budget, or on Bonsai with `--server http://127.0.0.1:8081` and
`--model-id Ternary-Bonsai-2-27B` instead of the provider flags, into a separate jobs dir):
```bash
cd ~/natlang-folder-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/inline-curriculum/handoffs-v1.ir.jsonl \
  ~/natlang/runs/inline-curriculum/handoffs-v1.jobs ~/natlang/runs/inline-curriculum/handoffs-v1.results.jsonl \
  --provider openai-codex --model-id gpt-6-luna --root-seed 909 --all --workers 1 --context-tokens 16384 --max-turns 20 \
  --reasoning-effort low --execution-plans --reuse ~/natlang/runs/inline-curriculum/handoffs-v1.reuse.jsonl \
  --reuse-surfaces 8db6432281451c575b2d720c6ed120cd4a212d5a058139c9c71f1d6e6ab6c78e >> ~/natlang/runs/handoffs-v1.collect.log 2>&1 &
```
(The reuse file holds 57 handoff runs finished under the previous tool surface; the surface hash changed when inline
children's openings changed. A collector only reuses a finished result whose provenance matches, so after any change
to `src/native/*`, `src/scope-compiler.ts` or `src/environment.ts`, pass finished results with `--reuse` and the old
hash with `--reuse-surfaces`, or they are redone.)

## State of the data

| What | Where | State |
|---|---|---|
| train4 shard (967 cases) | `runs/inline-curriculum/train4.ir.jsonl` | 954 done; missing indices 0, 2, 4, 5, 10, 16, 30, 35, 49, 59, 64, 67, 68 (Bonsai range 0-74) |
| Bonsai results | `train4.jobs/` (128 results, incl. 36 in 150-449) | resume as above |
| Luna slices | `train4-luna-slice.jobs` (517), `-slice2.jobs` (264), `-slice3.jobs` (54) | complete |
| Luna redo of Bonsai rejects | `train4-luna.jobs` (6 of 10) | redo loop resumes |
| Handoff tasks (833) | `handoffs-v1.ir.jsonl`, `handoffs-v1.jobs/` (66 done) | paused |
| Folder batch (106 cases) | `runs/folder/gen1.ir.jsonl`, `gen1.jobs/` (30 done) | 1 Luna worker |
| Folder pilots | `runs/folder/pilot{1,2,3}.*` | done; pilot3 9/10, all admitted |
| Directory shards | `data/teacher/directory/{train,test}.ir.jsonl` (35 / 16 cases) | rebuilt with fixed families; test is held out |
| SFT set v4 | `runs/spark-lora/v4` (8267 turns, 3686 sequences) | built before corrections/pairs existed |

Collectors write `*.results.jsonl` only when they finish; before building from a run in progress, gather its
`*.result.json` files (one JSON per line) into a JSONL. For train4, the results are spread over four jobs dirs
(`train4.jobs`, `train4-luna-slice.jobs`, `-slice2.jobs`, `-slice3.jobs`) plus the redo `train4-luna.jobs`; gather all.

## What was built today (all committed on main)

- **Corrected variants** (`ts-host/src/teacher/corrections.ts`, `scripts/inline-curriculum/corrections.mjs`): where a
  call failed and then fixed itself, the same run with the fix made first, given the reasoning from before the failure.
  Checked by replay (control replay accepted, variant accepted, fix shows the same output, variant passes admission,
  reasoning does not name what failed). Only the fix is trained. v4: 159 variants from 276 sites.
- **Handoffs and preference pairs** (`src/teacher/handoff.ts`, `replay.ts`, `scripts/build-handoffs.mjs`,
  `scripts/build-preference-pairs.mjs`, `scripts/export-preference-pairs.mjs`): a teacher takes over a failed run at
  its failure; the teacher's decision there is preferred to the failed one. Replay matches calls by opening text or
  similarity; sites and pairs are checked by replay (the rejected response still fails in place). Handoff tasks are
  ordinary IR records carrying `handoff`; the collector replays the prefix. Old hard-state queue removed.
- **DPO trainer** `scripts/train_dpo.py`: continues the SFT adapter, reference = that adapter (log-probs precomputed),
  sides run one at a time with the exact split gradient (tested in `tests/test_train_dpo.py`, run in the
  `natlang-train` image). `--sft-split` holds out the SFT run's held-out programs.
- **`scripts/build_lora_sft.sh`** now also: runs corrections and appends the variant turns; writes `pairs.jsonl`
  (corrected pairs, plus handoff pairs with `NATLANG_HANDOFFS=runs.jsonl,...`) and `preferences.jsonl` (rendered).
- **Runtime fixes**: unawaited nl calls and dropped async callbacks no longer crash the host (callables mark promises
  handled; eval-realm rejections become host events); an inline child's opening says it is the judgment and that nl is
  not available (16% of Luna train4 runs hit the nested-nl refusal before).
- **Directory-reducer families** (the other agent built the runtime and six families; reviewed and reworked):
  agreement oracles (`oracle.ts`: `agreement` level, `semantics.files_oracle` with exact/rewrite/csv/counts),
  delegation required only from 100 files, real banking77 messages for `folder_mixed`, crisp label datasets only
  (sms_spam, sst2 whole sentences, banking77, clinc_oos), exact label names, CoEdIT drafts that need the edit (no
  paraphrase task), CUAD's own non-compete definition with quoted sentences, HotpotQA token-F1 with a short answer.
  Admission finds children by evidence too and matches JSON-escaped evidence. Luna pilot 3: 9 of 9 finished runs
  admitted (was 1 of 6 in pilot 1).
- **File tool surfaces** for a probe: `--file-tools all|editor|files` on the collector (`native/prompt.ts`), recorded
  as `provenance.file_tools`. Default (`all`) unchanged.


## Restart after shutdown (latest operational state, 2026-09-29)

Shutdown was explicitly requested by the user. Do not schedule or restart generation until
work is resumed. No computer power-off command was issued; the user handles power-off.
All exact supervisor/collector/monitor PIDs were verified gone and Docker reports no running
Bonsai container. Completed data and JSON checkpoints were validated and the files flushed.

### Durable state

- `/home/werg/natlang/runs/shutdown-20260929-112702/checkpoint.json` contains exact stopped commands, queue/runtime/journal paths,
  active entries, pending counts, child PIDs (historical), saved metrics and SHA256 checksums
  for all5 active-case result/partial artifacts. Journals and current static manifest are
  copied beside it. `git-status.txt` records unrelated uncommitted work.
- Both journals have an intentional `shutdown` event, **not a finish/failure**. This keeps
  interrupted entries eligible to resume and does not increment Luna's provider-failure
  streak. Continue existing journals, never create a fresh journal to replay all attempts.
- **Bonsai unfinished batch:** `bonsai-four-20260929:44b0142c749f6614`, indices52–55,
  runtime-v34. Index52 has45 saved turns;index55 has47. Index53 has a completed rejected
  result(20turns);index54 has a completed accepted result(11turns). Do not erase/retry
  the rejected result automatically. The one-entry recovery queue retains the full batch
  identity and members; collector resume skips its matching completed roots.
- **Luna interrupted case:** `directory-expansion-20260929:luna:537`, CommitPack,
  runtime-v37,3 saved turns. Continue queue-v26 using existing v2 journal and checkpoints.
- Pending filtered queues at shutdown: Bonsai queue-v29 has364 entries (some batches);
  Luna queue-v26 has98 single-case entries. These counts include interrupted entries and
  are not counts of new training decisions.
- Latest admission snapshot:421 teacher results/246 admitted/1,070 approved decisions/
  117 held/zero unlinked. Published static V9:1,306 expansion cases/3,536 approved decisions;
  combined8,531 unique decisions,3,651 directory reducers(42.80%),42 tree-edit decisions,
  zero rendered-pair duplicates/holdout overlap. The manifest references immutable v9
  payloads; use `data/teacher/directory-expansion/static.manifest.json` as authority.
  V8 preserved in `runs/generation-check-20260929-1055/published-v8-preserved/`.

### Resume order and commands

1. Inspect `git status`, `docker ps`, GPU/process state and `runs/SHUTDOWN_LATEST`.
   Preserve concurrent working-tree changes. Frozen runtimes v34/v37 remain immutable;
   verify their `frozen-runtime.json` identities before use. Do not launch the stale
   `v37.rollout-config.json` or signal its old PIDs; its monitor was intentionally stopped.
2. Recreate the Bonsai server with the same4 slots/cap and memory settings. No student
   training has been authorized. Last launch command:

```bash
docker run --rm --name natlang-bonsai --gpus all --memory 6g --memory-swap 6g \
  -v /home/werg/natlang/vendor/prism/bin:/prism:ro \
  -v /home/werg/natlang/models:/models:ro -e LD_LIBRARY_PATH=/prism \
  -p 127.0.0.1:8081:8080 natlang-prism-runtime \
  /prism/llama-server -m /models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -fa on -c 53248 \
  --cache-type-k q4_0 --cache-type-v q4_0 -np 4 --kv-unified \
  --no-mmap --cache-ram 1536 --metrics --jinja \
  --chat-template-file /models/templates/Ternary-Bonsai-2-27B.jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --reasoning-budget 1024 --no-webui
```

3. Finish only Bonsai's interrupted batch with its original v34 runtime, preserving
   prompt/replay provenance. The supervisor exits after this one-entry recovery queue.
   Then use filtered queue-v29 with v37, retaining the same journal so finished batch/member
   identities are skipped. Do not run old and new Bonsai supervisors simultaneously.

```bash
python3 scripts/run_bonsai_queue.py /home/werg/natlang/runs/shutdown-20260929-112702/bonsai.resume-current.jsonl \
  runs/bonsai-recovery/journal.jsonl --runtime runs/bonsai-recovery/runtime-v34 \
  --case-seconds 600 --model-concurrency 4
# Only after the one-entry recovery supervisor exits:
python3 scripts/run_bonsai_queue.py runs/bonsai-recovery/queue-v29.jsonl \
  runs/bonsai-recovery/journal.jsonl --runtime runs/bonsai-recovery/runtime-v37 \
  --case-seconds 600 --model-concurrency 4
```

4. Resume exactly one Luna worker on v37/queue-v26. This includes its interrupted case;
   partial checkpoints replay under unchanged prompt provenance. No separate fresh repair
   worker is needed. The reviewed newline retry already succeeded before shutdown.

```bash
python3 scripts/run_bonsai_queue.py runs/luna-repair-20260928/queue-v26.jsonl \
  runs/luna-repair-20260928/v2.journal.jsonl --runtime runs/bonsai-recovery/runtime-v37 \
  --case-seconds 900 --provider openai-codex --model-id gpt-6-luna \
  --model-concurrency 1 --execution-plans --reasoning-effort low
```

Run long-lived supervisors with durable logs/session handling as before and record their
new PIDs. After resume, check fresh checkpoints and admission, then continue hourly
quality checks. QASPER live generation remains paused pending an extractive-equivalence
oracle; static document-read decisions remain useful. Source holds stay pending; never
rewrite original labels based on model judgments. Fifteen SciFact holds were root-verified
from the Luna sweep; details/corrections are under
`runs/generation-check-20260929-1055/luna-quality-sweep/`. Priorities after resume: monitor
these interrupted cases, verify v37 pagination behavior in fresh graph cases, inspect
new rejections, and improve QASPER oracle/source coverage before unpausing it. Maintain
25% reducer target across the final admitted mix and three ad hoc NL layers per file root.

## To do, in order

1. **Finish train4**: Bonsai's 13 cases and the redo loop. Then build the next Spark set:
   gather all train4 results into JSONL files, then
   `NATLANG_TS_HOST=~/natlang-runner/ts-host scripts/build_lora_sft.sh spark runs/spark-lora/v5 <train3, train3.long30, simple-v1, train3-luna, ref-v1, ref-composed-v1 results> <train4 gathered> <train4-luna> <folder gen1 gathered>`
   (refresh `~/natlang-runner` from main first: it lacks corrections/pairs). Add `NATLANG_HANDOFFS=` with gathered
   handoff results for handoff pairs. Check the build log's corrections and pairs counts.
2. **Bonsai folder pilot** once Bonsai's train4 range is done: run `data/teacher/directory/train.ir.jsonl` (or a
   dozen cases from it) with Bonsai from `~/natlang-folder-runner`, inspect with the approach below. Bonsai is the bulk
   teacher; it may fail differently from Luna.
3. **Tool-surface probe** (the user wants the shape our models do best with): run the same small folder cases with
   `--file-tools all`, `editor` and `files`, each with Bonsai and with the untrained Spark student (serve its GGUF
   with `scripts/serve.sh`, not beside Bonsai: the GPU holds one). Compare acceptance, admission, tool errors and
   turns per surface; then decide the surface and, if it is not `all`, make it the default.
4. **Train Sharp-Spark** on v5 (only when the user says so): `scripts/train_lora.py` with the Spark options used for
   v4 benchmarks (`--model models/candidates/spark-x25-4b-train --trust-remote-code --load-in-4bit --no-kbit-upcast
   --optimizer paged-adamw-8bit --max-len 8192 --batch-tokens 8192 --target-modules q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj`),
   then optionally `scripts/train_dpo.py` on `preferences.jsonl` with `--sft-split <run>/split.json`, and evaluate
   both on the held-out sets (simple/probe, and `data/teacher/directory/test.ir.jsonl` via `score.mjs`).
5. **On-policy loop after the first Spark run**: collect Spark on held-out/train cases, `build-handoffs.mjs` on its
   failed runs, collect the handoffs with a teacher, build pairs, DPO again.
6. **More folder data**: after gen1, larger builds (`build.mjs --families folder_triage,folder_index,folder_edit,folder_find,folder_extract,folder_mixed --seed <new> --shapes N`)
   into the regular queue. Known model pain point to address: models interpolate whole files into nl template
   instructions (`nl\`... ${text}\``) instead of passing the file; the runtime could point this out.
7. **Directory-reducer plan leftovers** (`plans/DIRECTORY_REDUCERS.md`): browser worker, document views, external
   trajectory conversion, Workspace-Bench/MuDABench adapters; the file-tool collapse waits on the probe.
8. **Old pipeline tests** `tests/test_student_improvement_pipeline.py` and `tests/test_student_improvement_runner.py`
   already failed at their setup before today (stale recipe infrastructure); fix or retire.

## How to inspect a run and iterate

- Per result: `outcome.accepted`, `outcome.oracle`, `outcome.files_check` (score, failed items), `outcome.detail`.
- Admission: `node ts-host/scripts/inline-curriculum/admit.mjs RESULTS.jsonl --ledger L.jsonl` prints rejection
  counts; each ledger line has `reasons` and `notes`. The standard SFT builder includes correct direct and
  delegated runs; use technique notes for coverage rather than treating direct answers as incorrect.
- Pattern that worked today: pilot a handful of cases per family, read the root's calls and what came back, and ask
  whether the task, its oracle or the runtime is at fault before blaming the model. Rebuild shards after family
  changes (`build.mjs` verifies every reference by replay and writes no shard if one fails).

Oct1 recovery follow-up: first four recovered Bonsai results all exact identity/hash matches, three current admissions/21 approved decisions, one rejection, zero missing spool results/source errors. Health report05:28:05.966UTC and matching canonical spool supplement saved; next formal check06:28UTC. Review of supervisor changes and Python syntax parsing completed; live supervisors loaded the health-wait/backoff patch at launch. In-flight server loss can still fail the current batch; health waiting prevents subsequent unstarted batches being consumed. No automatic blind semantic retry.

### 2026-10-01 21:02 UTC — additional OpenRouter teacher

User provided key in `~/.config/natlang/openrouter.env` (private, outside repo); **explicitly omit `enforce_distillable_text`**, never echo/read key into tool output. Worker started via `scripts/start_openrouter_teacher.py runs/space-bunny-20261001/root-approved-worker-plan.json`. Check `runs/space-bunny-20261001/worker-status.json`, `journal.jsonl`, `worker.log`, and `launcher.log`; initial live PIDs 822917/822922. Singleton flock guards launch. To stop, SIGTERM launcher; it propagates to supervisor and collector, retaining partials. Do not launch duplicates.

64 existing exact approved CommitPack cases, additional teacher coverage, 51 Markdown + 13 YAML/YML, first 8 mixed pilot. Root v25 native gold/reference audit admitted 64 with 256 linked decisions; `root-reference-review.json`. Exact plan/hash pins, private credential loading, zero-price Stealth-only routing/no fallback, auto tool choice, wire seed omitted, low reasoning/8192 output tokens, one request, exponential transport/failure delays, pause after 3 consecutive failures, retirement stop October 5. Model catalog transport template Qwen is **not** a fallback model; wire ID remains Space Bunny. User-required omitted distillation flag remains absent. Native tools/source golds/admission gates unchanged.

Isolated runtime v25 manifest SHA `86177acfe927ae3a0cc5f55a6558b44f6ccf7068c3dc2f5cfaf2966919935602`, derived from frozen v24 with only provider controls and compact exact diff presentation. Dependencies remain shared but critical SDK actual files separately hash-pinned. Existing Bonsai/Luna runtime/queued handoff pins untouched. New supervisor creates missing log parents and ignores valid cleanup observations before identity validation. Canonical cleanup fix still needs safe integration after partial Luna handoff completes.

Authority registers `additional_teachers.space_bunny` + active journal. Main current Bonsai/Luna outcome totals do not count this separate queue automatically: explicitly join its result paths or report separately. Generated snapshot already discovers all completed `.result.json` under `runs/`, so accepted explicit-teacher train rows enter its next rebuild; do not claim a newly published training snapshot. The 64-case pilot does not alter the global source-case count.

20:33 manual sweep: +49 rows since 20:10, all admitted. Saved new-reject audit 17:09–20:33: 8 supported-gold model failures, no source or admission bugs; exact edits must preserve internal/EOF line breaks. Diff preview change improves visibility without byte normalization. Root changes/proofs in `runs/space-bunny-20261001` and `runs/generation-check-20261001-resume/manual-check-2035/`; detailed course decisions in GENERATION_DECISIONS.

### 2026-10-01 21:40 UTC — DGX Spark Horizon deployment in progress

User authorized SSH `dgx` (samewerg, Tailscale, batchauthworks). **Use Horizon, not Qwen if a download is needed.** Finalmodel `IFM/K2-Horizon-MoVA-36B-A4B-FP8`, revision `feffd71999eb06bfa2fbb8ad220059b694077457`,48.36GBweights. Download container **natlang-horizon-download-fast**, authenticated existingprivateHFcachetoken (`whoami`success, explicittokenpassed, do notprint/cachecredentialinrepo),HF_XET_HIGH_PERFORMANCE1/eightfiles. Earlier Qwen/Horizonnormaldownloadcontainersstopped; caches/containersretained. Do not start Qwen.

User explicitly wanted oldtrainingstopped: `sdkb-bgkit` stopped, restart=no,exit137after30sgrace; nofinalcheckpointsaveconfirmed. Oldcheckpointfiles/mountspreserved. Stopreceipt `runs/dgx-generation-20261001/training-stop/` andremote`~/natlang-remote/training-stop-20261001/`. Donotresumeoldtraining.

Localcampaign `runs/dgx-generation-20261001/`; remotebundle`~/natlang-remote/campaign-horizon-20261001/`; user systemdservice **natlang-horizon-generation.service** enabled/running **waitingfordownload**, notyetgenerating. Readremote`worker-status.json`, `bootstrap.log`, `journal.jsonl`; `systemctl --user status natlang-horizon-generation.service`; download `docker logs natlang-horizon-download-fast`. Bootstrap verifies fullweightsize/LFSsha256,frozen/runtime/bundlepinsbeforelaunching **natlang-horizon-server** (localhost8081) thenqueue. Stopgenerationwith`systemctl --user stop natlang-horizon-generation.service`; modelserveris separate, stopexplicitlywhenneeded. Do notcreate duplicates. Servicepausesonfailed download/server/hashcheck; troubleshootandreviewratherthanblindrerunqueues. SSHdisconnectdoesnotstopremoteuser service; Lingeralreadyyes.

Pinnedimage vllm-node666307...4771d ARM64vLLM0.29.1rc1.dev467, actualHorizonarchitectureanddedicatedtool/reasoningparsersconfirmedavailable. InitialTP1/4sequences/32768context/0.65memutil; nativecollector16384/highreasoning/temp1/8192output. NeedconfirmactualGB10modelinitialization and thenadmission/throughput beforeexpanding. Runtimev26manifestSHA `a909719d18a74d61d791508837f311971b4716c042bbcfc11e8fa72e480cc54f`;630filesverifiedonARM64Node22.22, standaloneexact-versionnpmdependencylock,64goldreplays0modelcalls; root64currentadmit444linkedreferencedecisions. No source/gold/prompt/admissionweakening.

64pilotcases32reducers32workflow,16fourcasebatches, exactexistingrootapprovedIR. **Additionalteacher comparison, not64newcases ortransferofactivequeueownership.** Oncepilothealthy, supplyfresh/exclusiveDGXwork; do noteditfrozenin-flightBonsai/Lunaqueueassignments.

Local importer/sync `scripts/sync_remote_teacher.py CAMPAIGN --loop --authority runs/generation-check-20260930-hourly/check.json`, initialupdatedPID846994 (checklive). Pullsevery45s. Staging `runtime-import-staging/` deliberatelyexcludedfromautomatictrainingdiscovery; exactassignment/model/controlsverifiedbefore rawbytespublishedto`imports/jobs/<sha>/result.result.json`, withcurrentadmit/rejectledger `import-ledger.jsonl`. UnexpectedassignmentheldasrawJSON. Bothsuccess/failureevidenceretained, noautomaticDPOlabels. Defaultgeneratedsnapshotalreadyfindspublishedcompatibleteacherresults; noactualnewtraining snapshotpublishedyet. Statuses/syncauthorityadditional_teachers.dgx_horizon registered; remotejournaladdedonlywhenexists. Sharedauthoritylockrequiredandhourlydeadlinepreserved.

**21:47 concurrency update:** user requested higher concurrency. Latest reviewed Horizon plan now has **16 aggregate requests**, four collectors × four slots (children included), and server max-num-seqs16. Same64cases, four disjoint16casequeues, isolated jobs/journals. Previous4slotplan retained at `root-approved-plan-concurrency4.json`. Latest sync initialPID848038; four journals registered when they arrive. Service was restarted while still waiting for download. Download container unchanged/running. See `concurrency-review.json` and latest root-approved plan, which supersede the4slot settings above. No actual Horizon generation yet. Owned/isolated checks passed; full mutable workspace type check is blocked by unrelated improvement/teacher.ts edits, excluded from deployment.

### 2026-10-01 22:54 UTC — first active 50-minute check and DGX recovery

User requested an active 50-minute sleep/check loop. First wait completed and agent inspected workers/rejections. This works while the agent turn remains active; the background monitor itself does not wake this chat. Authority mode label corrected to describe the actual arrangement. Latest scheduled mechanical audit at22:11UTC: zero blocking reader errors; eight Bonsai spool results exact matched/admitted,49 decisions. Manual read-only result audit since22:11 found86 new results,86 current admissions,zero rejects across current Bonsai/Luna assignment result files. Proof `runs/generation-check-20261002/manual-0050/admission-audit.json`.

Found Horizon download exited1 at22:19: Xet CAS file reconstruction response-body decoding error. Disk88GiB free, no memory/disk exhaustion established. Bootstrap correctly paused without consuming any queue. Resumed same authenticated container/cache, pinned model/revision unchanged. Remote transient systemd unit `natlang-horizon-download-recovery.service` runs campaign `retry-download.py`: up to3 recovery attempts,60/120second exponential delays, singleton flock, explicit exhausted status, restarts reviewed generation service for each attempt. No token printed/copied. Read remote `download-retry-status.json`; current first retry downloading. This controller does not survive reboot; underlying reviewed generation service remains enabled.

Fixed two observability bugs: monitor ignored additional teachers; now flags paused/failed/unreadable/stale sync states and finished assignments needing replenishment. Remote sync previously exited on paused deployment; now keeps observing recoveries. New monitorPID887885 and local syncPID888355; check actual live state. No generation worker or pinned runtime was restarted/modified by this monitor change. Current waiting DGX bootstrapPID124209 (historical).

Completed Space Bunny pilot audited:64results,57admitted,7rejected. Six returns include extraneous prose; two of these also altered internal whitespace. Seventh has correct return but old exact-oracle Markdown EOF mismatch (343vs344bytes). Treat seventh as a candidate for explicit versioned Markdown terminal-newline migration, not automatically admissible; preserve original gold/raw result. Admission reason `wrong_return` also summarizes file failures, so inspect `files_check` rather than assuming every such reason means a wrong value. Detailed byte-difference evidence retained in manual audit. Source semantic prompt clarified final reply is only the function value, without completion prose/Markdown unless required. This source clarification is for future reviewed runtimes; active frozen24/25/26 remain unchanged. Prompt deployment/replay review and Space Bunny next reviewed assignment remain outstanding. No admission relaxation, output repair, or invented DPO labels.

### 2026-10-01 23:46 UTC — second active 50-minute check

Manual audit since22:50:80new Bonsai/Luna saved results,80current admissions,0rejects; `runs/generation-check-20261002/manual-0147/admission-audit.json`. Scheduled23:11audit had0blocking reader errors,12exact-matched/admitted Bonsai spoolresults78decisions. BothLuna/Bonsai remain running.

Xet recovery attempt1 repeated same CAS response decoding failure after~42minutes. Stopped attempt2/controller before further retry churn; retained container/cache. **Current downloader `natlang-horizon-download-http`**, pinned sameimage/model/revision, explicit authenticatedtoken, `HF_HUB_DISABLE_XET=1`, HFdownloadtimeout60,4fileworkers. Supportedfallback perHFenvironmentvariable docs. Old `natlang-horizon-download-fast` stopped, preserved. Latest local/remote rootapprovedplan points toHTTPcontainer; originalplan preserved `root-approved-plan-before-http-download.json`. Recovery transientunit now `natlang-horizon-download-recovery-http.service`, same bounded3attempts60/120backoff. Fixed recoverycontroller bug: newly-created container hasExitCode0 before ever running, so require `State.Status==exited` as well; bootstrap already checked actualstate and never falsely generated. HTTPcontainer confirmedrunning/authenticated. Fullweight verification stillmandatory. Local syncPID888355 continuouslyobservespausedrecovery.

Replenished SpaceBunny with **64existing reviewed mixed cases**,32reducers/32workflow, from exact DGXpilotIR. Newcampaign `runs/space-bunny-20261002-mixed/`, launcher902097/supervisor902106, one concurrentrequest. Rootfrozen25nativegoldaudit64replays/64admissions/444decisions,0modelcalls; exactpins/privateauth/freeStealthonly/nofallback/no distillation flag remain. Frozen25 unchanged; futurepromptclarification notyet deployed. Registered successorjournal/status inauthority and retained old completedassignment metadata. This adds teacherattempts, not64newcaseidentities and doesnotreassign local work. Generatedsnapshotwilldiscoveracceptedresults atnextrebuild; no snapshotpublished.

### 2026-10-02 00:46 UTC — third active 50-minute check, reviewed replenishment

Sweep since23:44:183new results,172current admissions,11rejections; proof `runs/generation-check-20261002/manual-0237/admission-audit.json`. Mixed64SpaceBunny completed59admitted/5rejects (3file edits,2invoicebooleans). RootplusLuna audit `luna-rejection-review.md` and `luna-rejection-receipt.json`:2BonsaiMusique strings are evidence-supported equivalents (380 AD vs380; Warsaw City Council with Polish parenthetical). Keep rawfailuresheld/ineligibleDPO; source-pinned versioned equivalence migration remains needed.2TreeDST hierarchy mistakes,1Luna unjustifiedabstention,3invoiceboolean mistakes,3fileedit mistakes. Root reconstructed request replacement against originalinput: Vulkan/README goldsmatchexactly; initial Luna source/runtime suspicion was incorrect and report corrected. Model removed unchangedVulkanCurrentStatus/escaping; README shortenedfind missedtrailing source space. No gold/admission relaxation.

Luna1 predecessor fullyfinished271keys complete/exactexport, rootverified. Replenished **Luna1 256existing reviewed mixedcases**,128reducers128workflow; `runs/mixed-teacher-successor-20261002/luna-root-approved-plan.json`,workerPID917795. Same256assignedadditionalSpaceBunny teacher in `.../space-bunny/`, launcher918474/supervisor918475. All541oldLunacampaign sourceIDs and alreadySpaceBunnymixed64programIDs excluded; **no globalLuna freshness claim**,comparisonattempts onexistingapprovedIR. Goldreplays256,currentreferenceadmissions256,1316decisions separatelyonfrozen24and25,0modelcalls. Source/golds/active frozenruntimes unchanged. Sourcepromptclarification stillpendingfuturedeployment.

Prepared **disjoint192case Luna2successor**, `runs/luna-slot2-successor-20261002/root-approved-successor-plan.json`,goldreplays/currentadmission192,1188decisions. Isolated single-slot derivative ofreviewedhandoffcontroller,source syntaxchecked, allruntime621files/currentpolicy/queues/code pinned. ControllerPID919770 waits for exactcurrentLuna2queue complete/exactexports beforelaunch; rejects unreviewedfailures/drift/duplicatejournals. Registered plan_path/command/hash/launchrecord inauthority. This prevents idle afteroldLuna2queuefinishes. CurrentLunaauthority/state pointtoactualnewworker1andoldworker2; completedassignmenthistoryretained. MonitorPID887885/sync888355 continue.

HTTPHorizon downloadhealthy/running,cache32138047220bytesat00:37 (~32.1GB of48.36GBweights, incomplete/cachebytesapproximation);68GiBdiskfree. `natlang-horizon-download-http`, authenticated,4fileworkers, noXet. NoHorizonoutput/modelinitialization yet. Scheduled00:11audit0blockingreadererrors,20Bonsaispoolresultsalladmitted126decisions. Return to50minutesleepaftercheck.

### 2026-10-02 02:07 UTC — fourth 50-minute check, Horizon inference and compatibility fixes

Local sweep since00:37:433new rows,427current admissions,6rejects. SpaceBunny mixed256:251/256admitted; otherworkers176/177. `runs/generation-check-20261002/manual-0336/admission-audit.json`. Luna2 automatic192case handoff succeeded,worker938249; Luna1 remains917795; Bonsairunning. Replenished SpaceBunnyremaining192 at `runs/luna-slot2-successor-20261002/space-bunny/`,launcher940161/supervisor940165. Alsoprepared reviewed541workflow successor (existingcases additionalteacher,0reducers): `runs/space-bunny-workflow-successor-20261002/`,goldreplays541/currentreferenceadmissions541/5862decisions. New generic `scripts/start_reviewed_openrouter_successor.py` controllerPID943134 waits complete/exact predecessorexports+finishedlauncher, exactauthority binding andpins; rootapprovedrolloverplanSHA60017d...ffe73. Registeredmonitorcontroller. Retainfinal25percentreducermix gate; nofreshglobalcaseclaim. No distillation flag.

Horizon authenticated HTTPdownload completed01:25UTC; all48.36GBweight hashchecks passed. Firstserver loaded45.52GiBmodelmemory. Found andfixed2deploymentissues beforeanycomplete result: autodetectedOpenAIlistcontent broketemplate string concatenation; add explicit `--chat-template-content-format string`. Then assistantmessageswithmissingthinking failed officialtemplate. Rootfixed actual transportbug: streamingassembler previously dropped `delta.reasoning` while only preserving `reasoning_content`; nowpreservesboth (sourcechange +isolatedfrozen27). Templateoverride onlypermits empty absenthistoricalthinking; existingthink/reasoningstrings unchanged, nofabricatedreasoning. Oldservercontainers/logs/failedpartialspreserved, zero complete results inbothfirstattempts, allinfrastructure—not DPO negatives.

**Current DGX** runtime `~/natlang-remote/runtime-v27/`, manifestSHA **88248d54c605b33616e76c237707601612289743450c74b43ba14a5dae905d7d**,630files; onlysource/distchat-completion assemblerchanged from26. RemoteisolatednoEmitpassed;64goldreplays/currentreferenceadmissions444decisions0modelcalls; no prompt/tools/golds/admissionchanges. Source-pinned `horizon-chat-template-v1.jinja` inbundle,servermountdeploymentread-only, explicitstringcontentformat; warmcompilecachepersisted at `~/.cache/natlang-vllm-horizon/vllm` (missing/tmpcubincachewarningfellbacksuccessfully). Server `natlang-horizon-server`, reviewedgenerationservicebootstrapPID250230, workers250925–250928,16aggregatecap. Newjobnamespace **jobs/thinking-r3/worker-{0..3}**, journals `journal-worker-{i}-thinking-r3.jsonl`, latestrootapprovedplan. Priorplans/sourcequeues preserved. Do notduplicateoldfailedqueueattempts.

Bootstrap nowoptional **zero-inference `/tokenize` templatepreflight** includingOpenAIlistcontent, syntheticassistanttoolcallswithnothinking, andpriorreasoning. Receipt token_count235/requestSHA832d91...882698 at02:00:46; catchesformattingbeforeburningqueue. No softwaretests added/run; AST/JSsyntax,noEmit,nativegold/realartifactadmissionaudits only. Actualnewinference02:00onward:HTTP200,15–16runningrequests,~44–52aggregategeneratedtokens/s; savedpartialsprogress1–2turns andpreservedreasoningverified. Some validtoolreplieshavenoreasoning, provingemptyhistoryfallbackneeded. Nofurther400inrecentlogs. **Nooptimalconcurrencyclaim yet; await64pilotadmission/latency/cachepreemptionbefore32requests.** KVcache~28.76GiB,maxutil.65; GPUmemorymodelreport~49.24GiBweights/nonTorch. Originaltrainingremainsstopped.

Local sync rearmedPID**947011** (manualbootstrapstop hadexitedprevioussync); continuouspaused-observationfix remains. DGXauthority runtime/journals updated. Read`sync-status.json`,`thinking-compat-recovery.json`,latestbundleplan,andremote`worker-status.json`. Mainlegacyhealthtotals stillrequireexplicitadditionalteacherjoin. NoHorizonfinaltraining snapshotpublished; automaticimporterwilladmitcompleteassignedresultsandholdunexpectedidentities, rawhistorypreserved.

Luna/independentroot6rejectreviewconfirmed1invoicebooleanmistake,2returnprose mistakes,1customer-evidencefieldoverclassification,2supportedMusiqueequivalents. **Root implemented separatelyversioned2case source-pinned equivalences**: `musique-reviewed-output-equivalences-v2.json` + `musique-recent-output-equivalences.mjs`, wiredsourcebuilder. Acceptonlyfourreviewedobservedstrings(380AD/Edictclause; WarsawCouncilPolishparentheticalwith/withoutperiod) forexactfullIRdigest/evidence; primarygolds/v1registry/rawrowsunchanged. NewIRsuffix `:reviewed-output-equivalence-v2`. Proof `manual-0336/root-musique-equivalence-v2-review.json`:2nativegoldreplays,4acceptedrealhistoricalanswerchecks,0modelcalls. **Historicaltrajectoryconversion stillheld until linkeddecisions/provenance migration; notautomaticallyadmitted, notDPOnegatives.** Futurefullsourcepublicationmustcarryallcases; do notpublish2casepartialasreplacementglobalqualityversion. Currentinflightfrozenqueuesunchanged. Sourcepromptreturn-formatclarificationfromfirstsweepstillfuturedeploymentonly.

### 2026-10-02 03:13 UTC — fifth active 50-minute check and automatic replenishment

Horizon inference is healthy: 34 unique imported artifacts by 03:12, 32 current admissions / 2 rejects, no assignment holds. Zuul correct return but omitted one explicit replacement trailing newline (required internal blank line); root source.replace(find,replace_with) exactly matches gold, 562 actual / 563 gold bytes. Invoice f02120... returns true for explicit invoice despite criterion asking whether NOT invoice; reasoning visibly reversed question polarity. Preserve both raw failures; no oracle relaxation. Root proof `runs/generation-check-20261002/manual-0456/root-horizon-zuul-review.json`. DGX HTTP200, 6–7 active requests at ~21–24 aggregate generation tokens/s, no waiting requests / zero KV preemptions. Four-case batch barriers leave unused slots while slow cases finish; review rolling batches for NEXT assignment after pilot, do not mutate in-flight queues or claim concurrency optimal. Disk49GiB free. Sync947011 healthy.

Luna1 completed256/256 exact exports and automatic single-slot handoff launched **64 reviewed existing mixed cases**, PID987397, `runs/luna-slot1-successor-20261002/`; native gold64/current reference admissions64/444 decisions, zero model calls. Rootapproved plan SHAfb3897...9b4f1, controller987391 finished. Luna2 PID938249 remains running192queue. Prepared disjoint next **128 reducer cases per Luna slot**, `runs/luna-reducer-slot{1,2}-20261002/`, compared source identities against recent512case assignments. Additional teacher attempts on existing approved static IR; no global freshness/new case count claim. Slot1 128CommitPack/384 linked decisions; slot2 72CommitPack+56TATQA/599decisions, nativegold/currentadmission128each, no source/gold changes, frozen24 unchanged. Standbycontrollers988705/988706 await exact current predecessor queue exports, pinned artifacts, authority singleton. PlanSHAs353c4370...706f5a andc159a278...b2290. Monitorregistry/authority/state updated under sharedlock, hourly deadline retained.

SpaceBunny541workflow assignment still running. Prepared next **256 reducer comparison cases**, same new Luna128+128 pool, frozen25 nativegold/currentadmission256/983decisions. Campaign `runs/space-bunny-reducer-successor-20261002/`; controller989259, rolloverplanSHAac92cf8a...2c5c4, waits complete exact predecessor exports and finishedlauncher. Same freeStealthonly/no fallback/no distillation flag, one request, backoff and failurepause policy. No active runtime/prompt changes. Finalwholetrainingrecipe25% reducer gate stillpending; these queues add coverage, not published dataset.

Luna read-only review point-in-time03:12: **933 raw-unique new artifacts since01:36:18.907Z, 922admitted/11rejects/0reader errors** (899local named campaign trees +34DGXimports). Categories:2supportedMusiqueequivalents,2agent-trace structuredfield mistakes,1customer-service batchblocked despite five supplied files,5invoice document-type boolean polarity reversals,1Zuul explicit-edit whitespace miss. Review evidence directory `runs/generation-check-20261002/manual-0456/`; exact current source/golds retained. Musique historical linkedtrajectory migration remains held; do not make them DPO negatives. Future workflow prompt should explicitly bind true to stated proposition and false otherwise; preserve source criteria/labels and version any changed case identity. No software tests added/run. No new training snapshot published. Returning to active50minute sleep/check loop.

### 2026-10-02 04:42 UTC — sixth 50-minute check: stale TATQA contracts found and migrated

Read-only Luna sweep since03:12: **543unique artifacts,481admitted/62rejects**, no admission errors; `runs/generation-check-20261002/manual-0605/reject-trace-audit.json`. Breakdown4trace-observability,25CommitPack,31TATQA,2invoice. These mismatches do NOT all establish model mistakes. RootCommitPack review reconstructs source.replace(request.find,request.replace_with)==gold for all25;15correctfile/wrongcompletionprose,2MarkdownEOFonly,6filecontent,2bothprose+filecontent. Rawbytes/golds unchanged. `root-commitpack-review.json`. Lastmechanical04:13audit0readererrors/SpaceBunnyfinishedalert; root resolved by newqueue below, manualauthorityreceipt saved.

**Found pipeline gap:** latest directory staticquality-v25 still includes **218TATQA IR without already-reviewed evidence-scale/numeric display instructions**, plus one already-modern exact operation derivative. Earlier newreducer queues copied these legacy IR directly. Correct arithmetic often came back as explanatory strings/units while gold expects bare scalar; some units/proportion cases remain semantically unresolved. Luna reports `tatqa-rejection-review.md`, `tatqa-case-review.json`. Do not mark all31 as arithmetic errors or DPO negatives. Added source-only runtimeFailureReason guard `legacy_tatqa_numeric_contract`; real31artifact candidate-policy audit holds all31 (`legacy-tatqa-negative-hold-review.json`). **Canonical dist not rebuilt yet**: policy guard needs safe rollout/reviewed re-pins before next dataset/DPO build; active plan pins depend on current compiled policy. Source nativeprompt adds boolean-polarity clarification. No blanket numeric repair, target injection or tolerance added.

Retired **unstarted** oldLuna2 reducercontroller988706 underauthoritylock (recordretired_before_launch, registry archived); activeLuna2 predecessor untouched. Migrated56TATQA in new128case successor `runs/luna-reducer-slot2-contracts-20261002/`, newversionedIDs, originalgold/input/reference/source lineage unchanged. Nativegold/currentreferenceadmission128/599decisions. Newcontroller1019382 planSHA90b9cf...c441a0 completedhandoff; **Luna2PID1021981** nowrunningcorrected128queue. Luna1PID1011175 running128CommitPack. PreparedLuna1next256workflow additionalteachercomparison of SpaceBunny newIR, nativegold/admission256/1448decisions, `runs/luna-slot1-workflow-next-20261002/`, standby1026458 planSHA7d363ace...ccc79ff. No newglobalcasecount/freshnessclaim.

**SpaceBunny frozen28 deployed**, isolated from25 with only prefix clarification: finalreplyonlydeclaredvalue, booleantrueexpressesstatedpropositionincludingnegation, useexact parsed editrequest strings/preserveoutsidewhitespace. Threechangedfiles source/distprompt/declaration; manifestSHAd59841ffdd2dc267d98892ab252ddc17c8bcce18d527f2922d71ab268ff1516c. IsolatednoEmit/JSsyntaxpassed, nativegold/currentadmit256/1448decisions. Current `runs/space-bunny-next-20261002/`, 256reviewedworkflow cases(64eachsource), launcher1021416/supervisor1021422, rolloverplanSHA538482...96bb50. Prior256reducerassignment finished/exactexports preserved. Same freeStealthonly/no fallback/no distillation flag/one request/backoff. Newstandby `runs/space-bunny-reviewed-retry-20261002/`: reviewed256reducer **prompt-improved retry comparison**,200CommitPackunchanged+56modernTATQA, gold/input unchanged, nativegold/admit256/983decisions. Controller1026804 planSHAb4bc68c0...a5b403. No automaticcausalDPO pairing across changedprompt/caseversions. Finaltraining25% reducer gate remains mandatory.

**DGX Horizon**57unique imports/54admit3reject by04:24, noassignmentholds; still2activerequests/~10aggregate generationtokens/s at04:34. Normalstreamprogress, zeroKVpreemptions earlier. Pilotstillrunning, do notstartsuccessorconcurrently. Prepared **256 exclusive current DGX successor cases**,128reducers(43TATQA43Musique42SciFact) +128workflow(32eachsource). Sourceidentities exclude recentlocal/preparedassignments+pilot; these are existingstaticcases/newteacherattempts, notglobalnewidentities.43TATQA moderncontracts applied; originalprecontractbundle/plan retained. Nativegold/currentadmit256/1143decisions. Four4-slotcollectors unchanged aggregate16cap; **sixteen-case batches** rollfourroots internally to reducebatchbarrierunderfill; batchresourcebudget19200sec proportional tofourwaves, notlanguage deadline. Model/runtime27/template/server unchanged.

Campaign `runs/dgx-generation-successor-20261002/`, remote `~/natlang-remote/campaign-horizon-successor-20261002/`; newservice **natlang-horizon-generation-successor.service** installed/inactive, notenabled yet. ExactremoteplanSHA544ce8e69c82c6c153ed5234427016a87864065f948b3da5f555c3f1e98bb353 verified. Genericroot-reviewed `scripts/start_reviewed_remote_successor.py`, controller1020275, rolloverplanSHAa2bc00f7...baff803, waits pilotfinished+all4journalsexactexport+64assigneduniqueimports+oldserviceinactive beforestartingnewserviceandreplacingsyncauthority. Reusesrunningmodelserver; no duplicatecollectors. Source importer fixed hardcodedruntime-v26: newassignment explicitly names local frozen `import_runtime`, originalpilotfallback unchanged. Afterhandoffreview, retireoldenabledservice/enablecurrentservice deliberately; reboot stillrequires explicitstoppedserverrecovery perbootstrap. No studenttraining resumed.

**Prevent recurrence:** new streaming `scripts/prepare_generation_ir.mjs INPUT NEW_OUTPUT` applies reviewedTATQA contracts, preservesgolds/input/reference/sourceidentity, exclusive/atomicnewoutput, produceschecksum/change receipt and says NOTreadyuntilnativegold/admission/materialization/pinsreview. Exactalready-reviewed signedP&Lderivative preserved only byfullIRSHA07a23b35...31016c; unknown derivedsuffixes failclosed. Whole-fileHFstaticrowread hitNode stringlimit duringnewsource selection; switchedselection tostreamedJSONL.

**Full static migration candidate staged, NOT published**: `runs/static-contract-migration-20261002/` carriesall1212originalcases,218contractchanges/994unchanged, no gold/input/source loss. `candidate.sealed.static.results.jsonl` uses canonicalreferenceRow/programRow to rebuild218native references withfreshcorrect runidentity/prompt+tracehashes/sourceconversionparentlinks; original994references retained. `sealed-reference-review.json`:1212currentadmitted/3198approveddecisions,0modelcalls. Earlier `candidate.static.results.jsonl` is onlymanualreplaycandidate witholdprovenance; **do notpublish it**. Unit/proportion semantic review stillrequired beforefullstaticversionpublication/cataloglink/defaultrecipe refresh. Luna subagent ongoing source-unitreview of31rejects, outputsmanual-0605/source-unit-review.*. Originaldefaultstaticquality-v25/recipe-v31 unchanged; no newtraining snapshot. Nextcheck priorities: finishsemanticunitreview, source-pinnedholdsifneeded, publishFULLstaticreplacementwithlineage afterreview; rollDPOguardcompiledpolicy safely; checkDGX automatic256handoff androllingutilization, reviewedretryquality. No testsadded/run. Returnto50minutesleep afterthischeck.

### 2026-10-02 05:56 UTC — seventh active check, explicit partial repair and supervisor fixes

Horizon pilot exhausted queues with **63/64 exports, 60 current admissions/3 rejects**, not64complete. One TreeDST (`inline-curriculum:source_treedst:66f656cc3fc3b9479762:v1`, pilotindex3) hit4800sec collection resource budget, with12 savedrootturns and repeated model verification/forbidden-loop errors. No server corruption established. Original timeout/partial preserved, no DPOnegative/positive invented. Root receipt `runs/generation-check-20261002/manual-0734/horizon-partial-repair-review.json`, failedfinish canonicalSHA0f2af40b...a15d, immutablecheckpointSHA5d2c717e...1d81. Old handoff correctly refused incomplete export; retired its failed controller/record. New remote successor helper requires exact reviewedfailedfinish hash and exact missing-program set againstoriginal64IR; accepts63exportsonlywiththat explicit accounting, records paused review status on failure.

DGX new assignment **257cases =256reviewed newteacher attempts+1exact-provenance partialrepair**,129reducers128workflow. Nativegold/currentadmission257/1145approveddecisions,0modelcalls. Repair keeps original source/index3/seed202610013600/frozen27/prompt/requestcontrols; runtime checkpointcopy separate from immutablearchive;14400sec repairresourcebudget. Newcampaign `runs/dgx-generation-successor-20261002/`, rolloverv2 plan/launchrecord retained; oldservice disabled,newservice enabled,running modelserver reused,aggregate16cap unchanged. Started05:49UTC. Startup caught real supervisor bug: batch16 plan exceeded inherited hardcodedmax5 guard, causing3collectors to exit before any case started. Controlledservice stop preservedcheckpoint/originaljournal; isolated reviewedrun_queue ceiling16 and bootstrap immediately pauses on nonzero supervisor exit; restart05:52 and all4collectors confirmedrunning05:54. No duplicate collectors. Sync had correctly stopped on stoppedservice; new syncPID1067655 restarted. Assignment/remoteplan hashes re-pinned, all before-fix artifacts retained. Current isolatedserver/runtime/model unchanged.

Canonical `scripts/run_bonsai_queue.py` now has explicit `--max-batch-cases` (default5,local-only,upper64), validates every queue entry before any collection; request semaphore remains independent. Existing localworker processes unchanged. Canonical `scripts/remote_teacher_bootstrap.py` now stops siblings and reports `paused_supervisor_failed` immediately for nonzero worker exit, preventing partially dead campaign from masquerading as healthy. CurrentDGX isolatedbootstrap hassamefix; isolatedqueue ceiling16 reviewedspecifically.14 completedhandoff controllers archived underauthoritylock before canonicalcode maintenance; actualactiveworker authority retained. ASTsyntax reviewed,no testsadded/run.

Luna2 migrated128 queue finished all128exports; new**256workflow** `runs/luna-slot2-workflow-next-20261002/`, workerPID1065760. Cases72invoice71customer42security71agenttrace, nativegold/currentadmit256/1510decisions. Luna1PID1043026 remainsrunning256workflow. SpaceBunny reducerretry256 finished05:44; new additionalteacher comparison ofLuna2same256 workflow in `runs/space-bunny-workflow-next2-20261002/`, launcher1066292/supervisor1066296. Frozen28 gold/currentadmission256/1510decisions,one request/freeStealthonly/no fallback/no distillation flag/backoff unchanged. Existingstaticcases/newteacherattempts, no globalfreshidentityclaim. BonsaiPID778887 still supplied. Luna subagent read-only audit ofSpaceBunny frozen28workflow/reducer retries ongoing; outputsmanual-0734, nextcheckinspect.

Sourceunitreview corrected afterrootaudit: TATQA000210/220 unsupportedmillion were modelerrors underexisting visibleunitcontract;000236 suppliedvaluationallowance note anchorssame table scale million (163.6delta matches797.8-634.2). **No newholds for those3**. Only000206/235 need prompt-only fraction/precisionclarification, originalgolds/input/referenceunchanged. Agent draft `manual-0605/tatqa-prompt-migration-draft.json` hasfullIR/source/input/goldhashes, NOTpublished. Rootreviewaddendum supersedesearlieroverbroadholdrecommendations. Full1212staticsealedmigration candidate stillNOTpublished untilpromptdisplayreview/nativegold/sourceproof; defaultquality-v25/recipe-v31 retained. Source-only legacyTAT numericDPOguard stillpendingcompile before nextDPO/datasetbuild; activequeue runtimes remainfrozen. No automaticnumeric targetinjection, broad whitespace relaxation, or causalDPOpairing acrosschangedprompt/case.

Continue active **50-minute sleep/check loop**, preserving absolutedeadline if subagentnotification interrupts sleep. BackgroundmonitorPID887885 hourlyobservations do notwakechat. Next priorities: SpaceBunny audit, DGX full16rollingutilization/partialprogress, fullstaticmigration publication and source-onlyDPOguard compiled rollout with review, source-pinned twofractionpromptvariants; keep final25% reducer mix gate. No training snapshot or finalstudenttraining published.

### 2026-10-02 06:53 UTC — user-directed model UX, Qwen/Gemma comparison, and DGX storage

Read-only SpaceBunny frozen28 review512rows:478admitted34reject, workflows256/256; reducerretry172/200CommitPack and50/56TATQA.28filegolds reconstructedexactrequestedreplace;14finalreturnerrors,13whitespacebytes,1badrequestfieldreplaceWith vsreplace_with. TATQA one false admission found: invalidtyped eval return was neverstaged but countedasprematuredecision beforeevidenceobserved. `curriculum.ts` now ignores explicit toolobservations refusingaresult-type;33firstDecisionfacts corrected across634savedrows, **exactly1admissionchanged** (000229 recoveredvalidcase). Typecheckpassed; audit `manual-0734/admission-maintenance-delta.json`,no testsadded/run. Canonicalcompiledcurriculum/policy rebuilt; prior source-onlylegacyTat numericDPOguard **nowactive**,31/31reallegacyrejectsheld (`legacy-tatqa-compiled-negative-hold.json`). Activefrozenworkerfilesunchanged;14completedhandoffcontrollerpinsalreadyretiredbeforecanonicalmaintenance.

Horizonquality through06:14:111imported,95admitted16reject (pilot**59/4**, successor36/12), originalmissingTreeDSTpartialstilltracked. Priorhandover60/3pilotcount waswrong; correctedhere. Report `manual-0734/horizon-quality-review.*`. Severalactualarithmetic/formulaselection/errors, negatedbooleanerrors, finalprose/protocol mistakes; Musiqueequivalence requirescase-by-casesourcereview, notautomaticrelabelling. Agentinitialinvoiceconflict claim wasincorrect: questionaskswhetherdocumentisNOTinvoice, rawinvoice supportsfalsegold; reportcorrected. SciFactclaim71 remainsinterpretationreview, no newlypublishedhold/rewrittengold. Treehint groundedearlierpilotpartialtrace, not onecompletedadmittedTreeDSTrow.

Sourceprompt addsinspection-vs-return distinction, exactoperands/units/formula directioncheck, directknown-tree-pathverification; sharedfinalvalue/booleanpolarity/exacteditprefixalreadystrongerthanHorizon27. Isolatedprompt-only **frozen29** prepared `runs/horizon-prompt-review-20261002/runtime-v29`,manifest2434e10141750b74333325ca7bbed9344f710a3117952d3e7e0534479eec2af2,3changedpromptfiles/630verified;257nativegold/currentreferenceadmission/1145decisions. It includescurrentpending scoped-folderpromptclarification fromparallelworkspacework; deploymentstillreviewedfutureonly. Neverresume27partialswith29. Prepared64matchedpilotIR gold/currentadmit64/444decisions in `runs/dgx-horizon-prompt-comparison-20261002/`; **noHorizoncomparisonservice/controllerlaunched**. UserthenprioritizedQwen3.6trial; reusematchedcasebundleforQwen, preservesHorizoncomparisonstage.

Twoexactsource-pinnedproportiondisplayvariants nowwiredin directorysourcefactory and`prepare_generation_ir`,module`tatqa-proportion-display-reviewed.mjs`: fullIR/input/sourcehashes, requested-share decimalround2/empty scale formatting, **no targetnumerals injected**, unchangedgolds/reference/source. Native2admitted/17approveddecisions,0modelcalls; `manual-0734/tatqa-proportion-review/`. Full1212modernstaticIR carriesallcaseswith2additionalchanges in`runs/static-contract-migration-20261002/cases-with-proportions.ir.jsonl`; preparedSHA b7ba2f7a6e2dbe72eb2fde1c99b73d5e505b2f6ecb59444793ddedd46b78eaa4. Fullsealedstaticpublication/catalog/defaultrecipe refresh remainspending; no two-casepartialpublication. Recipe-v31 stilldefault, finalstudentrender25%reducerspending.

**Speed correction:** do notdivideBonsai`llamacpp:n_decode_total` bytimeandcallittokens/sec. Itcounts llama_decode calls, potentiallybatched. Initial6xclaimwithdrawn. Actual30sec`tokens_predicted_total`sampleBonsai14.64aggregate t/s vsrecentHorizon30–33aggregate t/s,roughly2x currentworkloadthroughput, notmatchedsinglerequestbenchmark. Bonsaialreadyuses4casebatches, mayunderfillattail; previousclaimone-rootqueuewasincorrect. Considerfuturelargerrollingbatches under4requestcap afterreview. Horizonnewrolling16casebatches/all4collectorsrunning, no extraGPUmodel yet.

**Backgrounddownloads authenticated explicitHFtoken**,no credentialtextprinted. New`scripts/download_reviewed_model.py`: fullrevisionpins,5GiBdiskfloor,HTTPfallback4workers,modelLFS size+SHAchecks, durablestatus. QwenFP8 **Qwen/Qwen3.6-35B-A3B-FP8**,rev95a723d08a9490559dae23d0cff1d9466213d989,37,463,662,160weightbytes/42files. CurrentDGX unit/container`natlang-qwen36-download-20261002`,internalHFcache deliberatelyretainedsinceuserholds.cachemove. Metadataonlycachefoundinitially; actualweightsdownloading. Docker memorylimited4G, noGPU. Gemmauserrequestresolvedtoinstruction-tunedNVIDIABlackwell **nvidia/Gemma-4-26B-A4B-NVFP4**,reva19cfe00be84568a6867111c9a68c9c44fdcffe6,18,788,485,588weightbytes/2files (originalGoogleIT51.61GB alsoaccessconfirmedbutnotqueued). InitiallystartedlocalPID1089749, thenuseridentifiedexternalHDD: stoppedlocaldownloader, preserved/copiedpartial, nowDGX`natlang-gemma4-download-20261002` writes **/mnt/external/hf-cache**,authcachemountedreadonlyHF_TOKEN_PATH=/auth/token. Localpartialpreserved, no2simultaneousGemmawriters. Exactstatus dirs `~/natlang-remote/qwen36-download-20261002`, `gemma4-download-20261002`.

RootinitialstatementDGXlacksroomwasincomplete: onlyinternalcachefilesystemchecked. **External/mnt/external879GiBfreeinitially**,3.6TiBext4/dev/sda1, userwritable. Existing**QwenNVFP4** at`/mnt/external/bgkit-data/models/hf/Qwen3.6-35B-A3B-NVFP4`,threeweights23,424,338,320bytes, config/tokenizerpresent. DifferentquantizationfromrequestedFP8; LunaSSHpreflightsourceprovenance/LFSverification/currentimageflagsongoing (`runs/dgx-model-comparison-20261002/qwen-deployment-review.*`). NoQwenserverorGPUtrialstarted; Horizonstillrunningwhilebackgrounddownloads/copy. AimtryexistingNVFP4soonandFP8afterdownload, explicitlylabelquantizations; pairedqualityandthroughputbenchmark, thenreviewteacherchoice. PinnedvLLM image remains666307...471d,currentCPUarchitectureARM64.

**DGX disk cleanup authorized:** homeusage sdkb-runs155GiB, bgkit-data-nvme146GiB, .cache131GiB; Dockeroldbuildcachepruneuntil24h reclaimed**7.693GB** (log`runs/dgx-disk-cleanup-20261002/build-cache-prune.log`), internalfree~49→55GiBbeforenewdownloads. No container/images/volumes/trainingcheckpointdeletion. Userfirstauthorizedall3moves, thenexplicitly**held.cachemove**: moveonlysdkb-runs/bgkit-data-nvme. New`scripts/relocate_reviewed_dgx_data.py` exact2sources, externalmountcheck, knownwritablefdpause, rsync-aHcopy/checksumdryrunwithzerochanges, switcholdpathtosymlink, removeonlyverifiedoriginalcopy; durablelogs/receipts, pausesonmismatch, no.cacheentry. RemoteidleI/O/nice10/2GiBunit **natlang-storage-relocation-20261002.service**,PID455694, currentlycopyingfirst155GiBtree into`/mnt/external/natlang-storage-20261002/sdkb-runs`; then146GiBbgkit-data-nvme. Status`~/natlang-remote/storage-relocation-20261002/status.json`, notcompletedyet. Originalpathsremainliveuntilverifiedcutover. Do notresumeoldtrainingwhilecopying. Finalclaimspacefreedonlyafterreceipts/cutover/deletion; ~301GiBexpectedeventually, notalreadyreclaimed. cache/modeldownloads/serverpathsremaininplace.

Currentsteeredtaskcontinues: complete2treemoves/checkstatus, authenticated2downloads, QwenNVFP4preflight/trialsoonwithoutduplicatingGPUservers, preserveallHorizonpartials/queuedIR/results/authority. Resume50minutesleep/checkcycleaftercurrentnecessarysetup; model/storagejobsrunindependently. Keepgenerationworkerqueuesfilled andauditnewrejections. Respectparallelworkspacechanges inprogramimprover/self-improvement scripts/tests/authored-source/catalog; rootmuststageonlyownedhunks.

### 2026-10-02 07:04 UTC — Qwen NVFP4 selected; FP8 download stopped

User explicitly accepts the existing NVIDIA Qwen3.6-35B-A3B-NVFP4 checkpoint and cancels the FP8 download. Stopped DGX transient `natlang-qwen36-download-20261002.service` (Restart=no, no boot enablement); its downloader container has exited and been automatically removed. Incomplete FP8 cache is retained, and `.cache` stays in place. No further FP8 download or comparison is scheduled. NVFP4 full three-shard SHA256 verification matches `nvidia/Qwen3.6-35B-A3B-NVFP4` revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`; deployment/pilot preparation continues. Gemma external-cache download and the two authorized data-folder moves continue.

### 2026-10-02 07:07 UTC — Qwen NVFP4 server startup initiated

Horizon successor collectors stopped cleanly under systemd; preserved their existing jobs, journals and checkpoints, all four exited -2 in stopped status. Disabled its boot enablement while another model owns the GPU. Horizon server stopped but retained. Confirmed no GPU compute processes before starting `natlang-qwen36-nvfp4-server` at localhost8082, exact verified NVIDIA revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`, pinned image ID `sha256:666307b207f9f583f6048111e90c5d177355a2f94d9101494c8c85c5b277471d`. Config: context32768,16sequences,8192batchedtokens,.45GPU utilization, FP8 KV, FlashInfer attention/Marlin MoE, qwen3 reasoning/qwen3_coder tools, language-model-only/string chat content, external compile cache. Container startup launched; health/tokenization/tool smoke and64case pilot still pending, no Qwen quality claim yet. Local command receipt `runs/dgx-model-comparison-20261002/nvfp4-server-command.json`; agent prepares separate64case frozen29 pilot. FP8 stopped receipt `fp8-download-stopped.json` and remote `~/natlang-remote/qwen36-fp8-download-stopped.json` overrides original root-owned downloader status (which still says downloading). Gemma and twofoldermoves continue.

Qwen startup course change 2026-10-02T07:13:00.957775+00:00: initial safetensors mmap loader made only ~1.3GB disk-read progress while large copy competed for HDD. Stopped/retained initial container as `natlang-qwen36-nvfp4-server-mmap-startup`, restarted same pinned model/config with `--safetensors-load-strategy prefetch`; temporarily froze storage relocation. Safety timer `natlang-storage-copy-resume-20261002.timer` thaws in20minutes even if agent interrupted; thaw earlier on readiness. Prefetch completed33.19seconds and first shard loaded51.68seconds. No model/source/precision changes.

### 2026-10-02 07:20 UTC — Qwen pilot and fresh Space Bunny queue running

Qwen NVFP4 healthy; synthetic string/tool-history tokenization372tokens, parsed eval smoke84completiontokens/53reasoning tokens returned real tool_call (`eval`, code `6 * 7`). No smoke output published as training. Weight loading/prefetch/kernel tuning completed; resumed storage relocation and canceled thaw fallback timer, FreezerState=running. Four Qwen pilot supervisors462243–462246, bootstrap462222, unit `natlang-qwen36-pilot-20261002.service`;64gold-reviewed cases,16aggregate requests, frozen29,19200resourcebudget. Root caught initial isolated bootstrap missing remote bundle directory before any collector/model cases; copied nested bundle then restarted. Fixed planned nested journals to rootjournal names for existing sync includes. Generic `sync_remote_teacher.py --teacher-key dgx_qwen36_nvfp4` supports separate authority key, syncPID1105784. Local campaign `runs/dgx-qwen36-nvfp4-pilot-20261002/`, imports0initially, collectors durable progress confirmed. Historical Horizon pilot used frozen27; Qwen uses29, so historical results are a same-case baseline with prompt confound, not a strict isolated model comparison. No quality claim until native imports/rejections reviewed.

Space Bunny predecessor256finished; Luna requested pool had256/256 identities already attempted by SpaceBunny, so prepared256new SpaceBunny identities from reviewed workflow quality-v4,64each fourfamilies,182sourcegroups,0overlap among1877historical unique teacher IDs. Nativegold/currentadmit256/1678decisions,0unlinked. Rootverifiedallpins and approved/launched `runs/space-bunny-workflow-next3-20261002/`, launcher1105410/supervisor1105415 running. Frozen28/one request/free-only/no distillation unchanged. Source/golds untouched; pendingnativeadmission eachnewoutput. Bonsai and two Luna workers still active.

### 2026-10-02 07:35 UTC — All DGX generation switched to Qwen NVFP4 by user choice

User explicitly selects existing NVIDIA NVFP4 for ALL DGX generation, clarifieslocalworkersunchanged, and asksforrejection-driven improvements. No further controlledmodelcomparisonsplanned. Durable authority `dgx_generation_policy` binds model/revision/endpoint16cap; `dgx_horizon` retired into`retired_remote_assignments`, containers/rawresults/checkpoints preserved. Root exactIRhandoff receipt `runs/dgx-model-comparison-20261002/horizon-to-qwen-handoff.json` confirms **257/257 Horizonqueuedcases** arecoveredbyte-equivalentlyasJSONbyfresh Qwenpilot64+mixed256; no cross-model partialresume.

Pilot completed64exports,**55admitted/9wrong_return rejects**; fourbatchfinishreportssum95418completiontokens/569.58sec≈167.5aggregatecompletiont/s, excludesstartup. Current broader256campaign (`runs/dgx-qwen36-nvfp4-mixed256-20261002/`) nativegold/currentadmission256/1143decisions,128reducers128workflow, unchanged frozen29/source/golds/model/controls16cap. Unitnatlang-qwen36-mixed256-20261002, bootstrap463563/supervisors463589–463592 running, sync1111255 separateauthoritykey. Rootcaughtplannednestedjournalsagainandfixedroot journal-worker-i.jsonl forsync; isolatedbundleallpinsverified beforeapprove. Initial~59imports46admit13reject; nine carryunreviewedMusiqueextractiveequivalence markers, readonlysourceauditpending; pilotnine-file/tree/invoice auditpending. Next512reviewed queue preparation assignedLuna,256reducers256workflow disjointfromcurrentQwenIDs; no launchuntilnativegold/pins/rootreview. Bonsai/twoLuna/SpaceBunnycontinue unchanged. Gemmadownloadandtwo-foldermovescontinue; .cacheunchanged;FP8cancelled.

### 2026-10-02 07:43 UTC — Rejection audit and future prompt30

Detailed evidence `runs/dgx-qwen36-nvfp4-pilot-20261002/qwen-pilot-rejection-review.md/.json`: four CommitPack wrongfiles reproduce exactly from recorded model edit arguments; suppliedrequest literalreplacement equalsgold in allsevenfilecases. Three otheredits exactgoldfiles butplaintextreturnwrapper/prose. Oneinvoice literalNOTpredicate mistake; oneTreeDST dropssixsiblingnodes after26loop/recursion/pathdiagnostics. No edit-runtime/parser/sourcegold bug established; rejectedrowsnotpromoted.

Sourceprompt now clarifies actual return_result toolinvocation versus printedwrapper; directory guidance reads/parses editrequestfile andpasses exactpath/find/replace_with valuesdirectlytoeditText, avoids transcription/newlineboundarybugs, returnsrequestedpathratherthanassumesdone. Prompt-only `runs/qwen-prompt-review-20261002/runtime-v30`, manifest80a75e100a80be939e0b155e6cd0e51f50990b9f970207f2201b3dc39dd2d970,630files/3promptfileschanged; singleprompt TypeScript compilationpassed,0tests/modelcalls. Frozen29active unchanged. Next512nativegold/reference review30inprogress; deployonlyfreshqueueafterrootreview. Paralleluncommittedscoped-folder guidance retainedinworkingfile/inheritedfrozen29→30, rootstagesonlyownedtwohunks. Canonicalwholehostdistnotrebuilt, sharedworkpreserved. MixedMusiquealiasessourceauditpending; no blanketoracleloosening or exacttargetinjection.

### 2026-10-02 07:56 UTC — Two source disputes held; migration instruction withdrawal

Independent root read of complete matching sourcearticles confirms Apolloquestion based-on ambiguously refers tostatue type orsupposedbronzeoriginal; LaHavequestion needscommunitydistrictlinkbut suppliedWaldenparagraphonlylocatesWaldenontheshoreofLaHaveRiver. AddedpendingSourceReview ids`2hop__124719_533073` and`2hop__167674_776856`, preservedgolds/input/history, noSFT/DPOpromotion pendingclarification. Isolatedsource-review TypeScript compilepassed; onlycanonicalsource-review.js/.d.tscopied, nofullhostbuild. Realtwo-artifact currentadmissionreceipt `runs/qwen-source-review-20261002/admission-receipt.json` shows source_review_pending+wrong_return forboth. No selectednext512 matchesnewholdIDs; newproof/pins being refreshed. LegacyMusiquecontractpreparationgap found: existing source-pinned aliases omittedby scripts/prepare_generation_ir.mjs (itpreviouslyhandledonlyTATQA). Agentextendshelperinexistingreviewedsourcebuilderorder, first1212candidate17rowschanged, fullsource/gold/input/referenceinvariants preserved, nativechanged-rowproofpending. No active256IRmutation.

User withdrew no-backward-compatibility/no-state-migration request becauseitwasintendedforanotherproject; no suchplanrestriction wasadded. Continuearchivedcorpuspreservationandreviewedtraining-data transformationsunderexistingauthorization. Qwen allDGXpolicy remainsactive;256mixedbatch nowcomplete, next512rootreviewpendingpin/proofrefresh tokeepgenerationgoing.

### 2026-10-02 08:02 UTC — Reviewed MuSiQue transformations restored to IR preparation

`scripts/prepare_generation_ir.mjs` now carriesalready-approved source-pinnedMuSiQue alias additions/removals, Oklahoma contract, v1/v2 outputequivalences, andminimum-age contract alongsideTATQA. ExplicitbaseIDs/fullIRhashpins, registrysourceproofs, allowed-mutationprojection andunchangedgold/input/reference/sourceinvariants; exclusiveoutput+durableversion2receipt. Entire1212modernIR carriedover,17caseschanged;native17/17gold/currentadmission/materialization,0sourceconversionerrors/0modelcalls. Fullcandidate `runs/generation-check-20261002/manual-0734/generation-ir-reviewed/cases.v2.ir.jsonl`,SHAc6e6bf8c03387c96ddced978ca442bd7461255a412a49160268f1faa1dcbba16. Rootcaughtminimum-agecomposedvariant idempotencybugduringcode review; nowfullvalidatorrecognizesalready-reviewedfinalvariant beforealias-onlyguard, tautologicalIDcheckremoved. Actualfull1212v2→v3 transformation0changes/0transforms/identicalbytes, existingnativeproofstillmatches. No testsrun, fullstaticpublication/catalogrefreshstillpending. Next512queuegetsmoderncontracts beforelaunch. GemmaNVFP4downloadfinishedandall2weightfilesSHAverified at07:55:33; cacheexternal, notdeployedbecauseDGXallgenerationQwenbyuserchoice. sdkb-runs stillchecksumverifying, nooriginalspaceclaimedfreedyet.

### TextWorld reference follow-through — 05:03 UTC

Reference source now uses explicit finite iteration limits, surfaces failed playable progress, and returns the computed live certificate with eval finish:true instead of a literal expected value. Commit recorded by root separately. Versioned fixture `runs/textworld-iterate-reference-repair-20261003/candidate-v2/reference-audit.json`, SHA `e351f3c619130d1a28aa407319a4053ff6f0bfcfbe4ec2f2dc64b65576b63f37`: x64v39 actual native playable done/7approved decisions after three real actions; missing-object case blocked/quiesced with1helddecision, no positive target. This fixture uses oldv39 plus action evidence; combined strengthened-grader replay and refreshed fullworld/ARM proof remain required before world launch. Old cachedIR and gold preserved.


## 2026-10-03 20:35 UTC — memory recovery and continuation

The owner authorized autonomous pauses. Qwen’s completed queue left its server
holding about 86 GiB of unified GPU memory. It was stopped cleanly (container
retained, restart policy `no`). CUDA free memory then measured about 90 GiB.
Advisory `POSIX_FADV_DONTNEED` over selected pipeline files recovered another
~11 GiB of Linux free memory: 97,479,589,888 → 109,347,905,536 bytes. This is
file-cache reclamation, not deletion or a reduction in live model weights.
Receipt: `runs/neuralese-integration-20261003/memory-receipts/cache-reclaim-development-v2.json`.
Current DGX readout: 88 GiB free / 117 GiB available; Qwen remains paused.
Luna v46 (two workers), Space Bunny v46 and the local full Muon training remain
active. No generation queue, source data, gold or training checkpoint was removed.

Prevention:
- `common.parquet_rows` reads 128-row batches instead of materializing a complete
  row group. Protected-data Parquet and Arrow scans now also convert at most 128
  rows to Python at once; Arrow’s underlying IPC batch remains source-defined.
- Finalization caps boilerplate group sets at the three-group classification
  threshold and releases indexing maps before subsequent passes. Classification
  and split policy are unchanged; the compact dedup/closure index still grows
  with the number of records.
- Reference serving and GGUF export memory-map trainer checkpoints, so unused
  optimizer tensors are not eagerly copied into RAM. These entry points require
  normal modern torch.save checkpoint files; no legacy fallback was added.
- Enabled DGX `natlang-development-cache-hygiene.timer`: every ten minutes it
  advises release of selected pipeline cache only when MemFree is below 16 GiB.
  No global drop_caches, sudo, data deletion, or active process termination.

Deployment: bounded readers and finalizer changes are on DGX main; already
running Python stages retain imported code until their next stage/process.
No extra tests were run for this memory-only follow-through. Evidence consists
of the recorded live memory readings, successful reclamation receipts and active
service checks. Do not interpret Linux MemAvailable as CUDA allocation capacity.

Other continuation state: private fork https://github.com/werg/llama.cpp-neuralese
(branch `neuralese`, `8d302c6a2`) preserves the DGX WIP and server fixes; CUDA
port parity passed four cases. Non-streaming HTTP/TS end-to-end verification is
still pending. S3 phase D completed in a preserved child run; held-out correct
payloads beat shuffled payloads modestly, but BF16 cache agreement was 2/4, so
investigate that before scaling or launching E/F. Full S1 conversion and held
v13 compiler migration are running on DGX. Candidate outputs require review and
execution replay before training admission. Restore Qwen and the reviewed v5
successor after the remaining GPU checks; do not treat its paused server as live
generation. Older sections above describe the initial handover, not current state.

### 2026-10-03 20:44 UTC — cache diagnosis, faithful checkpoint deployment and generation restore

Same checkpoint, four held-out prefixes and fixed payloads were compared on DGX.
Snapshot fork vs unforked readback logit difference was exactly zero in every
variant. Float32 reference arithmetic matched all four eight-token continuations
(max logit difference 3.05e-5). BF16 fast with/without the convolution kernel each
had one mismatching case; BF16 reference had one different mismatching case.
Every mismatch was a full-forward top-logit tie. This supports numerical drift
rather than snapshot corruption on these inputs; it does not establish universal
cache parity. Evidence: `runs/neuralese-integration-20261003/cache-diagnostics-report.json`.
Harness reports now include per-case mismatch steps, margins, token agreement,
dtype and kernel identity; no tolerance or admission rule was relaxed.

Reference serving now infers trained cutoff/block maximum, rejects incompatible
explicit values, restores phase-F LoRA tensors, and retains float32 head parameters
like the trainer. Default remote base loading pins the intended revision. LoRA
alpha metadata now follows the actual 2*rank trainer setting. Phase-D checkpoint
loading was exercised by the diagnostic; phase-F restoration still needs its own
end-to-end verification after that phase exists.

Generation restart exposed a real identity bug: the importer hardcoded
assignment.json while authority could bind assignment-v2.json. Future sync/import
uses the exact authority-pinned assignment, hashes its bytes once, and rejects a
changed file. Preserved v4 ledger remains associated with its original assignment.
The handoff accepts this only through a pinned predecessor import-assignment path,
verified supersedes SHA, and an exact operational projection differing solely in
native proof attestation metadata. No ledger rewrite or case-result relabeling.

Qwen container restarted; reviewed v5 controller
`natlang-qwen36-v5-restored-controller-20261003.service` waits for actual model
readiness before launching 1024 cases and retaining the importer as its child.
Controller SHA a09121936a0c77f64bd145b0fd9b129bb3b2572c540efbf518e132d8c8e69a0e.
Original v5 proof/assignment/preparation are preserved; v2 artifacts explicitly
normalize verified report fields and bind their originals. Worker/model/prompt,
source/gold/protected policies and concurrency256 remain unchanged.

SWE and agent conversion finished raw stages; full dedup/closure finalization is
underway. Full v13 migration remains held pending replay. Their heavy disk readers
were briefly frozen for Qwen weight loading, with independent two-minute thaw
safety timers. Finalization now includes the existing writer-target leakage check
rather than leaving it only in a separate validation CLI. The already-imported
finalizer must still receive a separate validation pass before admission. This
check is a heuristic with documented short-answer/exact-reference exemptions,
not complete causal proof. No candidate corpus was published.

Local full Muon run reached checkpoint6580/13165; fixed held-out loss at6500
was0.832901 (6000:0.836124),128examples/no skips. CPU execution eval at6000 is
still processing actual HTTP requests, not yet a completed score. Luna v46 two
workers and Bunny v46 journals show continuing case progress.

### 2026-10-03 20:55 UTC — verified Qwen recovery and narrower cache hygiene

Qwen endpoint became healthy; controller launched, but runner rejected setup
before any cases because root changed the native review to v2 without rebinding
its authorization SHA/path. This was an integration error, not model failure.
Preserved original approval artifacts, setup-only worker/pool status and one
pool_setup_failure journal event in remote `setup-failure-v2-preserved/`.
Verified no jobs/results/case events/pool-plan existed. Corrected authorization
v2 and wrapper v3 passed actual ARM `--preflight-only` for all1024 identities.
Root recovery receipt `pool-v9-alternates-v5/root-setup-recovery-approval-v1.json`
binds their exact hashes. Restarted the inactive v5 unit with wrapper v3.
Now worker state running (collector2719648, supervisor2719604), real Qwen POST
responses200, and retained sync/import child2711836 reports running/zero initial
artifacts/no import errors. No failed model samples were relabeled as positives.
Authority records the recovery. Controller readiness now also checks the
native-review SHA and both exact remote receipt paths in authorization, so this
mismatch will fail before any future launch.

Cold-start model page cache was reclaimed after readiness: DGX MemFree rose
from~1GiB to~23GiB, subsequently~16GiB as readers resumed; MemAvailable~34GiB.
A manual whole-mirror advisory scan blocked in kernel lock_buffer for several
minutes and was terminated; its original empty receipt remains evidence of that
interruption. No cause beyond the observed kernel wait was proven. Periodic
hygiene now scans only pinned dataset cache and Qwen weights, excludes recently
modified files (five-minute guard), and records in-progress receipts plus counts
instead of leaving blank files until completion. Data, output files, checkpoints
and model weights remain intact. Timer stays enabled every ten minutes with the
16GiB MemFree pressure trigger.

The full port run prerequisites are recorded in
`plans/neuralese/S3_FULL_RUN_HANDOFF.md`: unfinished E/F review, full corpus
quality/replay, bounded sampler, faithful base/export/serving handoff and actual
Muon support for a new port run. Existing port AdamW state is not silently
converted; current local main training already uses Muon. Current repo changes
are committed/pushed and copied to DGX. Resume requested work/check/sleep cadence;
a timer/controller does not itself wake the assistant session.

### 2026-10-03 22:00 UTC — requested sleep completed; first follow-up sweep

Completed the requested ~50-minute sleep and performed a live sweep. Main local
Muon checkpoint6940/13165; all main, eval, Luna and Bunny units active. Step6000
execution evaluation completed:23attempted,1policy-held,11complete successes,
11semantic failures,1incomplete; no contract/resource/infrastructure failures.
Semantic accuracy is11/22completed cases (50%), not11/24. Step6500 is evaluating.
Report `runs/training-periodic-eval-20261003/execution-eval-v6/execution-results/execution-results/step-06000/adapter/report.json`,
SHA f878cbb7c119a73033ff244659b2fb93f450ee9fe5a2f245a54b8722f996641e.
Do not claim a controlled improvement against older differently handled baselines.

Qwen observed347imports:340admitted/7wrong_return rejects (snapshot, not final
campaign totals). Current collection passed setup and serves real requests.
Luna finished237cases across two workers; Bunny395. Next v47 refill preparation
assigned to the existing Luna source-audit agent, using source-pinned Qwen v5
cross-teacher IR, native/protected/source/history checks and current frozen v41;
no launch/authority change delegated. Agent must report exact pins for root review.

Reviewed the first seven Qwen rejects against their exact criteria/state: answers
assert financial hardship/legal threat/recognized purchase/account anomaly or
other allegations without the corresponding required evidence. No confirmed gold
or runtime defect. Review artifact
`runs/generation-check-20261003/memory-followup-v42/qwen-rejection-review.json`
preserves source, gold and rejection hashes. Future prompt clarifies per-file
question/state isolation. Do not inject ordinal labels or accept these wrong
answers as training positives; retain repair/DPO lineage.

Found a real preview defect: long filename-stem keys can exhaust the value budget,
causing even a one-character string label to appear empty with a truncation note.
Native previews now retain strings up to16characters (shorter than the truncation
notice). Isolated TypeScript compilation passed in
`runs/generation-check-20261003/memory-followup-v42/compiled`; no tests/model calls
were added for this source follow-through. Active frozen runtimes unchanged; fresh
freeze/native proof still required before teacher deployment.

Full S1 finalizer is progressing, not stuck: its read counter advanced81→83GB,
CPU~80%,resident process~400MiB. Cgroup8GiB largely comprises~7.6GiBfile cache,
with memory-high events/no OOM. Preserve the live pass rather than discard its
work. Future streams now request sequential/NOREUSE kernel cache hints and log
pass1counts every100krecords. These hints are optional; record/split semantics
unchanged. Full v13 migration processed~92krows and is still writing its pending
candidate;27compiler-failed rows so far remain held. Its in-flight exhausted_input
was initialized true; source now reports null/running until the stream finishes,
then complete/exhaustion explicitly. This reporting fix does not alter active
candidate output or prove its completion. Replay/failed-row review remains due.

DGX~33GiBavailable/14GiBfree; cache timer succeeds, both conversion units active;
external179GiBfree. Home20GiBfree: do not pull the entire huge DGX raw corpus into
that space. Preserve remote outputs and return manifests/review receipts; arrange
an explicit storage projection for full corpus mirroring before claiming a home
replica. No data was deleted or excluded from the dataset catalog by this sweep.

### 2026-10-04 continuation: checkpoint identity and live sweep

Main full Muon run is active at step7040/13165,56320trained exposures,0skips.
Bunny481/512complete; Luna149+155finished, with one preserved incomplete
(refill-v46-luna2-0134). Its Codex request-processing error is infrastructure,
not a resource timeout; five replies saved, zero result rows. Keep failed/held
with an explicit retry obligation, never promote it as positive. Qwen, S1
finalization and full v13 migration remain active; DGX33GiBavailable.

Fixed phase-F pilot resume: retain the complete phase schedule while stopping
at individual review boundaries. Checkpoint optimizer groups now save/replay
LoRA layer order and exact parameter names before loading moments. Historical
LoRA checkpoints without group identity are preserved and refused for explicit
repair; pre-LoRA D checkpoints remain compatible. Require port config/schedule
and validate phase position. Earlier missing harness reports are not synthesized
from later resumed weights; existing reports are included in the summary.
Luna performed two read-only reviews; no tests/model runs were added. Actual
F resume/deployment exercise remains due before claiming full resumability.

### 2026-10-04 v47 continuity and rejection review

Bunny v46 completed512/512 with complete output accounting. Reviewed sequential
handoff succeeded: v47-r2 now running launcher2749345/supervisor2749346,512cases.
Root worker plan SHA4d11a51c0f0c6314a26e636056d34a51f4285649369b20d1379dbd7d3f756e5d;
rollover SHA00132143f99ff23a434ef058c46de6aeeaad5faab49489a649d44fcd226ed987.
Service natlang-bunny-v47-reviewed-rollover-20261004 retains ownership via waitpid.

Two Luna v47-r2 256-case slots are approved and waiting independently for v46
PIDs to exit, with exact terminal accounting gates. Plan SHA
58813f94e7964412ed66eeb778de81bc5052ebb638998ab2283e1f709cfebcf0, service
natlang-luna-v47-reviewed-slots-20261004. Failed v46 Luna2 key0134 remains held
and pending explicit retry. Its raw journal line hash c2c100b... differs from the
canonical JSON event hash a53c9c31e7acba347c5a67885d0a0cc58fa9bd8e6e1dc079b7a6bf2e9c0394ed
used by the handoff checker; root checked both against saved evidence.

All v47 tasks are exact Qwen v9-v5 parent records. Bunny512 equals the union of
disjoint Luna256+256; current native replay512admitted/2246approved decisions,
0held/denied/unlinked, source/protected/history proofs reviewed. Same frozen v41
is retained; this is cross-teacher source reuse, not new source coverage.

Qwen latest snapshot545imports=534admitted/11wrong_return; a later12threject
appeared during review. Four follow-up rejects were legitimate wrong answers:
open issues labeled resolved and later explicit clarifications ignored. Luna
review artifact runs/generation-check-20261004/qwen-followup-rejection-review.
No source/gold/admission relaxation. Future prompt now clarifies whole-conversation
corrections and distinguishes proposed remedies from success. Active frozen
runtimes unchanged; new freeze/native proof required before deployment. Main
step7000 heldout loss0.8185059633,128examples/0skips.

### 2026-10-04 post-sleep sweep and E/F continuation

Actual sleep22:18:46→23:11:49UTC (~53minutes including tool overhead). Main
full Muon run7360/13165,58880exposures,0skips; step7000loss0.8185059633.
Qwen v9-v5 finished1024:1001admitted/23wrong_return. All23source/gold rejects
reviewed; no source/runtime fault found. Exact duplicate question/state in two
composites is not independent evidence. Review-beyond-11 files under
runs/generation-check-20261004/qwen-followup-rejection-review.

Full language migration finished111301rows,35070changed,33failed rows,122340
annotations/3745capture sites. Output SHA8cfd43aad51f33c875b030a5d0c22ce2371e7bd6d529f378b63001eae5fb2648.
All remain held pending execution replay. Local small receipts/failed rows and
Luna independent review in runs/neuralese-integration-20261004/v13-full-candidate-v2-receipts;
full7.8GiB candidate remains DGX.33rows have36failures across7trajectories;
conservative compiler passes preserve code on diagnostic change, not proven
unsafe sources. Cumulative annotation acceptance/diagnostic diff logging may
improve completeness later, with new pinned candidate/replay.

Stopped idle Qwen after final imports, reducing DGXusedRAM~87→4.6GiB/free93GiB,
then resumed preserved D checkpoint1400 into fresh
runs/neuralese-s3-pilot-20261004-resume-ef-v1. First8GiBcap attempt OOM before
E's first update (204MiBallocation,80GiBphysicallyfree); preserved attempt log.
32GiBbudget restart completed E200/F300 atglobal1900; phase_index6/step0.
Actualoptimizer_lora_layers=[15,14,13,12], adapter_layers=[12,13,14,15].
No change to AdamW for this lineage; main full run remains Muon.

E stopping was nearly deterministic:197/200batch means16; three slight length17
continuations, no mean below16. Paired identical actions cancel their centered
REINFORCE score even when payload rewards differ. No sampling/log-prob mismatch
found in read-only review; no claim16is optimal. A diagnostic CLI now reports
per-position logits/probability/entropy on sealed prefixes without changing
policy. Future exploration changes must use matching behavior log-probability
and a new checkpoint lineage; current E/F result preserved.

Prepared Qwenv9-v6 next1024, actualARMv41 proof1024admitted/4333approved,0held/
denied/unlinked,58train groups,0protected/historycollisions; all v5 assignments
excluded. Still residual customer-service buffer, not balanced/new coverage.
Caught BEFORE JOBS preparation-contract errors: native child paths doubled
bundle/bundle; ad hoc hash refresh wrote manifest into preparation.json and
left stale manifest hashes. Original files/failure evidence preserved. Root
review packet-v2, native/auth/assignment/launch-v3 and preparation-review-v3
are authoritative. Root's initial preparation projection used the wrong screen
key; corrected to controller's prior_screen contract without loosening checks.

Approval helper now accepts an explicit versioned packet, registers v6
predecessors, emits canonical proof fields/child paths and derives its controller
preparation receipt directly from checked history. No hidden packet fallback.
Remote preflight-v2 passed1024,0jobs, no pool plan. Conditional config-v3 SHA
00b2777ef0445ba1e91d2aa22264a5fee8e89be15e1903dbff71f3b5e2914e92;
service natlang-qwen36-v6-reviewed-controller-20261004 waits only for Qwen
endpoint readiness and retains its sync/import child. Qwen restarted after E/F
completed. Check actual controller/collector readiness before claiming generation.

### 2026-10-04 full candidate audit and stopping evidence

S1 full conversion completed23:30:59UTC: final68GiB at DGXexternal
full-20261003/resume-20261003T1948-review1/final. No home68GiBreplica is claimed
(home19GiBfree). Dataset remains a candidate until independent current checks.
No actual freeze occurred when root tried to protect Qwen warmup I/O: the S1
service had already finished; the unnecessary thaw timer was canceled.

Explicit validate --schema now fails before scanning if jsonschema is missing,
instead of silently running only stdlib checks. The schema validator is cached
once per process; report binds the exact applied schema bytes. Luna made scoped
source edits, root reviewed; no tests were added or run. Full current schema/
leakage validation is next, with bounded streaming inputs and explicit held status.

E/F completed1900 and saved all4LoRAoptimizer groups in release order. CPU
stopping diagnostics exercised actual phase-F serving load (base/control/head/
LoRA restored), not HTTP/C++ deployment or optimizer-resume parity. Sealed4
prefixes stop at16: D early-stop probability~0.00093–0.00106; F~0.00111–0.00127.
CPU/BF16 diagnostics are limited to these prefixes, not population estimates.
AfterF correct-minus-shuffled NLL margins: extractive−0.1323, multihop−0.1238,
tooldigest−0.02228; cross-source cosine0.3887/rank95.17. Mixed sample metrics
(no universal performance claim); greedy lengths remain16. Local receipts in
runs/neuralese-integration-20261004/stopping-diagnostics-receipts. Fresh E
exploration trial should temper stopping with matching scoring and record
behavior config/lineage; preserve this completed baseline.

Luna v47 both active PIDs2770822/2770834 after exact per-slot accounting; Bunny
v47 active. Root has requested a single future Luna repair of preserved v46
provider-failed0134 after a current slot finishes, not an extra concurrent worker.
Qwen weights loaded; warmup in progress. Nextv6controller waits on readiness.

### 2026-10-04 resumed current full-candidate audit

Qwen v6 collector and local Luna/Bunny services remain active; main local Muon
training and periodic execution evaluation remain active. Full 68GiB S1 output
is held pending current independent checks, not counted as admitted training.
Audit uses a separate pinned .venv-neuralese-audit environment. Streaming reads
request optional sequential/NOREUSE kernel advice, without dropping global
cache or changing records. Audit dependencies are pinned separately from the
training environment. Schema, leakage and duplicate checks do not replace
source-policy review or execution replay of migrated teacher trajectories.

Current full audit launched as natlang-neuralese-full-current-quality-audit-20261004
on DGX, commit06eb254; MemoryHigh6G/MemoryMax8G/CPUQuota200%/Nice19. Input receipt
pins manifest and checker/schema/dependency hashes. Manifest SHA
2bd25a842abf8e715272cec15e221234a8200e6a4e5c7a083bc573645aa961f9.
Candidate has1,870,591 total records:1,361,548train/52,439validation/456,604test,
45family files; previous snapshot's1.36M count referred only to train. Current
independent audit must supersede converter's old static check. Protected hits
are moved to heldout by finalizer; do not interpret raw hit count as train leaks.
Keep source-policy/unknown-license holds distinct from schema validity.
Small local manifest/input receipts at
runs/neuralese-integration-20261004/full-candidate-audit-receipts.

23:51UTC: main Muon step7540/13165,0skipped; periodic fixed128subset loss at7500
0.8138042 versus7000 0.8185060. Bunny405/512finished; Luna v47slot1 18 andslot2
19finished. Qwenv6has239partial jobs,49updated within60seconds,HTTP200responses;
no completed imports yet, so active work is verified without claiming admissions.

### 2026-10-04 next Bunny queue gated and new failure review

Bunny v48 next512 exactQwenv6cases reviewed as parent-exact subset,512native
admitted/2162approved/materialized/0held/unlinked/denied,0same-provider exact
ID/payload collisions including currentv47assignments. Root independently
rechecked parent bytes,allpins and sevenhistoryinputs. Workerplan SHA
2d0ce42fe584c916d14b4a5978f7f67e4dbead943bb83080ab602f71fe7b6f9a;
rollover SHA05453635e8495295ba18ddf8a5cca5ec10db4f0ae3a65cddc21339edb09cb21f.
Service natlang-bunny-v48-reviewed-rollover-20261004 waits for exactcurrentv47
completion/accounting and releasedproviderworker beforehandoff. FreeStealthonly,
no distillationflag/paidfallback;cross-teacherbuffer,notnewsourcecoverage.

New Bunnyv47job436 e19c2c72eabae7e6 quiesced at40turns:manyemptyresponses and
malformed Minimaxprotocoltext/NUL,then incomplete resultobjects. Luna forensic
reviewrequested; preserve rawnegative/excludeddata,don'tstriptextintopositives.
Qwenv6first27results alldone;importerprocessing normally. Fullschemaauditscanned
11GiB by23:54UTC,processRSS~90MiB withservicecacheboundedby6Ghigh/8Gmax.

### 2026-10-04 00:55UTC actual cadence sweep

Actual50mininterval00:04:57→00:54:59UTC, interruptedbrieflyto review/commitLuna's
replydiagnostic,thenreturnedtosleep. No schedulerclaim. FullindependentS1audit
completed00:23UTC:1,870,591records/45files,0schema/structural/leakageerrors,
0duplicateIDs,6.0GiBpeak/0swappeak,26m30CPU/35mwall. SchemaSHA
6c2aa847c372c71c5800c3e502f125247cdae56c38e723aa2cf675b896495152.
Localreport atfull-candidate-audit-receipts/report.json. Remainingcandidateholds
are source/outcomepolicy and independentprotectedsplitreview,notformatmigration.

Bunnyv47completed512/512;v48automaticreviewedhandoffworked,189/512finished.
Lunav47slots105and109/256complete;one-caseretrycontrollerstillwaiting,notthird
worker. Qwenv6at529imports:515admitted14wrong_returnrejected. Luna reviewing
newrejects;rootrequestedlast319residualbufferpreparation,allactivev6assignments
excluded. MainactualMuonstep7880/13165,0skips/localGPU100%/7673MiBof8188MiB.
Fixedstudentexecutioneval7000completed8success/9semantic/1contract/3resource/
2incomplete/1policyheld;smallmixedoutcome,notuniversalqualityimprovementclaim.
Resourcecaseskeepresourceclassification;600sCPUevalcapisnotlanguageconstraint.

Bunnyjob436forensicreviewfound28emptyvisible/noactionreplies,sixmarker/NUL
responses,40matchedrequestpairs,0invalidstream/retry,twomissingfieldsreturns.
NoHTTP/SSEbodyavailable;cannotproveupstreamorigin. Keepincomplete,notpositive
orsemanticnegative. Futureobservationonlycommitted9fb4a46;v41unchanged.

### 2026-10-04 01:35 UTC deployment fix, split audit, and worker continuity

The independent full split audit passed 1,870,591 records: 985,434 source groups,
zero cross-split conflicts, and zero protected matches outside test. Its 81,471
protected matches are in test. Local receipt:
`runs/neuralese-integration-20261004/full-candidate-split-audit-report-v1.json`.
This does not settle source-policy, background alias closure, or teacher outcome
admission; no automatic promotion of the full candidate corpus occurred.

Phase-F checkpoint step 1900 exported successfully to DGX external storage at
`data/neuralese/exports/phase-f-20261004-v1`. Model GGUF SHA
56ba15f94301717a5e9ae9ed1a0f15310108346fb63b088ce2938a1d9463c8b0;
heads GGUF SHA a2c8facf7cdb27fab050eb3e28e6fefc03a75ad7c7f67f40ed67a5a5ccd5cdbe.
The first ordinary C++ HTTP response was empty. Diagnosis reproduced the failure
in Python by omitting BOS: the merged F model predicts EOS without BOS and `ok`
with it. The server now adds vocabulary-required BOS once before first prefill,
without duplicating an explicit BOS. Private fork commit 023131332 is pushed and
present on both hosts. CPU and CUDA server builds completed; actual HTTP checks
used CPU only. The same ordinary request now returns `ok` with 15 prompt tokens.
A controlled write/read request exercised a 16-position, width-1024 block and its
safetensors download. These checks establish only the exercised deployment paths,
not autonomous quality or full Python/C++ numerical parity. TS transport remains
unverified. Diagnostic server was stopped to release RAM; receipts are under
`runs/neuralese-integration-20261004/phase-f-http-v1`.

Bunny v49 root review independently verified all artifact pins, exact complement
to v48 within Qwen v6, zero history ID/payload collisions, and the native proof.
A stale runtime manifest hash in the proposal was corrected only in a new approved
plan; originals are preserved. Approved worker SHA
6bd382145613c033290055ef6bbe8d9837f5bb9b8743284aed842a185dabc2ad;
rollover SHA c0dd266b0c174035c6f617bf2824f314da1fb6f603ed4618f93a9995647ba7bf.
`natlang-bunny-v49-reviewed-rollover-20261004` is waiting for exact v48 completion
and released worker ownership. It does not add a parallel provider worker.

The 32-case balanced single-source wrapper prototype retains exact input/gold and
source metadata, eight cases per workflow family. Initial prompt tokenization fits,
but full file-observation token budgets, native admission and fresh history review
are still being prepared. Do not promote initial-prompt fit to a full budget proof.
Main local Muon training reached step 8040/13165 with zero skips; the fixed 128-case
held-out token loss at step 8000 is 0.8045394. Execution quality remains separately
measured. Qwen, Bunny and both Luna workers remain active.

### 2026-10-04 01:37 UTC remaining reclaimable cache fixed

The timer covered raw datasets and Qwen weights but omitted the completed 68 GiB
final corpus and immutable phase-F export. Added those two exact directories to
both the checked-in systemd template and deployed DGX service. No mutable queues,
training outputs, unfinished finalizer state, broad volume roots, or global cache
flush were added. Previous deployed service saved in the local DGX receipts.
The existing age guard and per-file POSIX_FADV_DONTNEED remained unchanged.
An actual cleanup increased DGX MemFree from approximately 3.6 GiB to 19 GiB,
reduced cache from 31 GiB to 15 GiB, and left Qwen generation active. MemAvailable
remained about 33 GiB and swap about 414 MiB. This reclaims clean file cache;
active Qwen model/KV allocations remain. Future finalized immutable snapshots
must be added explicitly when their audits create substantial cache pressure.

### 2026-10-04 01:39 UTC trained TypeScript handoff exercised

The existing DGX compiled `neuraleseServerModelTurn` and `NativeToolAgent` completed
a two-turn controlled session against the trained phase-F C++ CPU server. The
server-written literal became a typed `Neuralese<string>` return reference; local
and remote stores agreed on its content ID and 16 x 1024 payload (65,536 bytes).
Two transport exchanges were recorded. Receipt `phase-f-http-v1/ts-handoff-report.json`
pins the compiled artifacts used. This is controlled transport/runtime verification,
not autonomous output quality or complete numerical parity. Diagnostic server was
stopped immediately afterward; Qwen serving continued. No test suite was run.

### 2026-10-04 01:39 UTC Qwen rollover confirmed

Qwen v6 finished all 1,024 imports: 998 admitted and 26 rejected. The reviewed v7
controller passed terminal/import/pin checks and launched the exact 319-case
successor automatically at 01:39 UTC. Remote collector service is active and the
root sync child is retained. Rejection review now targets the final 12 new failures
beyond the earlier 14-case snapshot. Bunny v48 and both Luna v47 slots are active;
Bunny v49 and one-case Luna retry controllers remain gated behind their exact
predecessors. Full S1 source/outcome inventory and balanced wrapper observation
budget/native proofs are delegated as read-only/preparation work. No candidate
holds or gold criteria were waived.

### 2026-10-04 final v6 rejection review and future prompt clarification

The terminal 1,024-row ledger has 998 admitted and 26 wrong_return rejects.
The final 12-rejection review found explicit sibling-file evidence contamination
in five artifacts, rather than a file-read/runtime/gold defect. Root inspected the
saved legal-threat example: matched input contains only a delivery-delay inquiry,
but the model explanation borrows a lawyer threat from a sibling file. Existing
future prompt source already prohibits cross-state merges and reconciles later
corrections; it is not present in frozen v41. Added a narrow keyed-answer check:
bind each output key to its matching file and verify evidence from that file.
No gold/criteria/admission changes, forced delegation, or label-specific hints.
Active v41 queues remain unchanged; a new freeze and native/source checks are
required before this clarification is deployed. Final review receipts are under
`runs/generation-check-20261004/qwen-v6-rejection-review/final-snapshot`.

### 2026-10-04 02:30 UTC cadence sweep and corrected source expansion

Actual interval 01:40:26 to 02:30:28 UTC, with a brief rejection-review interruption
followed by return to sleep. Qwen v7 completed 319 imports: 306 admitted and 13
semantic wrong returns. All 13 are reviewed separately; file/runtime checks pass,
with one recovered filename typo. Bunny v48 completed 512 and v49 is generating.
Both Luna v47 slots subsequently completed 256 each; the gated single repair
started in slot 1. Root is preparing independent successor slot plans.
Main local training was at step 8360/13165, zero skips, GPU fully occupied.
DGX cache remains under control: around 21 GiB free, 13 GiB cache, 412 MiB swap.

Full S1 outcome aggregation passed exact file/manifest/protection identities across
all 1,870,591 records. Labels: 983,324 gold, 95,292 checked, 790,876 teacher,
1,099 failed. In the train split, gold+checked total 654,999; teacher 705,705;
failed 844. Labels are not synonymous with final source-policy admission. Optional
missing upstream revision metadata is not itself a blanket exclusion reason.
Exact source/input identity and background alias review remain distinct checks.

Sealed v42 has only seven source/compiled changes relative to v41: per-file/key
and whole-conversation guidance plus bounded parsed provider diagnostics. x64
manifest 8593e73a210d64f21ddf21008a83d68435078e20da500c9d981334d5845f3e8c;
ARM manifest 3a507ea98deb4c7fd8f1c1ff4fd1e04dff15495c4476b44a91ae48ef9001ecb1.
ARM local and deployed physical closure checks passed (20,706 files, 22 internal
symlinks). Old dependency versions/admission implementation are retained; TypeScript
compile passed, no test suite run. Active queues have not been changed in place.

New balanced one-source directory supply is 860 cases, 215 per workflow family,
limited by security source eligibility after train/protected/type checks. These
are new task variants of prior sources, not fresh source coverage. Full observation
budgets fit context 32768 with actual 8191-token response reserve; worst total
22475, margin 10293. Initial planning-only budgets were insufficient for seven
invoice samples at context 16384, so they were not treated as launch proof.
The first pool draft invented unsupported curriculum.slice=folder_one_source;
runner join caught it before launch. Failed v1 is preserved; v2 preserves the
source's valid slice and is still awaiting final native runner/root approval.
Do not claim admission-only proof establishes runner identity validation.

### 2026-10-04 continued handoffs and provenance guard

Root independently rehashed all 25 Luna2 v50 pins, rescanned 11,722 pinned
history rows with zero exact ID/payload collisions, joined all 512 source and
native records, and rechecked predecessor terminal accounting. Slot 2 launched
under natlang-luna-v50-slot2-reviewed-20261004 with frozen x64 v42. Slot 1's
separate repair finished; its outcome and successor binding remain under review.
Bunny v49 continues on its unchanged v41 runtime.

Balanced Qwen v10-v2 passed root approval-helper recheck and remote ARM v42
preflight (860 cases, no jobs started by preflight). Exact source question/state/
gold and protected train closure remain preserved. New ctx32768 accommodates
full observations; 256-request serving remains unchanged. Canonical source helper
review caught a duplicate declaration before commit. The pinned v2 input uses
old v3 wording only in its adaptation description despite v4 task identities;
future helper prose should be accurate, without mutating pinned launch inputs.

Future finalizations now hash exact normalized JSONL inputs during the existing
streaming passes, verify all three passes match, and record bytes/rows, protection
identity and scoped finalizer code files. File stat/hash changes abort before
publishing output. This does not infer upstream release revisions or historical
converter code, change record contents, or rebuild the old 68 GiB candidate.
The audit confirms null optional upstream metadata alone is not an exclusion.
Background source alias closure remains a separate unresolved proof requirement.

Main full local Muon run reached step8500/13165 with GPU100%, not a new pilot.
Heldout token loss improved to0.80454 at8000, but the tiny CPU execution sample
does not show corresponding monotonic gains. Some lost successes are resource
caps, and two inspected cases have actual semantic/file-output regressions.
Preserve checkpoint comparisons; do not stop or declare success from loss alone.

Luna1 repair terminal accounting passed independently; slot1 v50 now launched
with a separate 512-case v42 plan. Both workers have disjoint exact parent
subsets and zero prior same-provider ID/payload collisions. Qwen v10 has a
root-approved controller, remote preflight and launch request. Handoff checker
now accepts OS aarch64 and Node arm64 terminology; source-preserving selection
proofs without a redundant payload digest derive it from exact pinned IR with
proof ID/index join and an explicitly pinned digest helper. Separation is still
checked against every prior campaign and within the new queue. No proof, gold
or source bytes were modified to make checks pass. Canonical wrapper reproduction
compared all860 rows: only adaptation description differs(v3historical→v4accurate).
Retained data exact background-source overlap scan approved separately with
bounded resources; no exclusions applied from missing provenance metadata.

### 2026-10-04 03:56 UTC cadence check

Actual50minuteinterval03:05→03:55, briefly interrupted to repair a poorly
performing source overlap audit. Original SQLite design generated heavy page/index
churn: stopped safely and preserved receipt+DB. Root review caught a reducer
Counter/int bug in replacement before launch. Corrected pinned digest-only
external-sort scan now progresses under1CPU/Nice19/2Ghigh/4Gmax/idleIO,
no record changes or exclusions;1.5Mrows staged,68GiB exact source files under
review. Final input/output/protection identities remain mandatory.

Main Muon run8800/13165,70400trainedexposures,0skips,GPU100%. Step8000
CPUexec eval completed11success/9semantic/3resource/1policyheld,0infra.
Qwen v10 imported166:165admitted/1wrong_return,remote170complete/860,
no error files. Root inspected rejection: accepting pending refund proposal
incorrectly treated as resolved; correct gold and file/key binding, no system
defect found. Luna v50 slots66/75complete of512 each; Bunnyv49 at429/512.
Refill preparation and detailed rejection/eval review assigned to existing Lunas.
DGX~33GiBavailable/16free/18cache,swap412MiB; cache timeractive.
Current main commit911c338 pushed and DGX checkout fast-forwarded. No newtest
suites; runtime proof/preflight, actualpipelineaudits and liveworker accounting
used for decisions.

### 2026-10-04 refill and concrete quality issues

Bunny v49 completed all 512 cases with full output accounting. Root independently
rehashed v51 pins, joined its 430 exact v10 source and native records, rescanned
9,461 provider history rows with zero ID or payload collisions, and verified the
512 latest predecessor finishes. V51 now runs with 32K context and frozen,
standalone supervisor and launcher files. Free Stealth routing and omission of the
distillation flag remain. Original 16K drafts are preserved. Optional queue context
controls are validated before any case starts. The fixed calendar expiry was
removed in favor of live compatible free endpoint checks, as the user requested.
Existing loaded workers retain their prior code; future restart plans need reviewed
pins. Historical supervisor source remains in Git.

The conservative source audit matched all 45 file hashes and 1,870,591 rows in
1,841 seconds. No exact training source matched a heldout question or target.
Context overlaps remain: 1,230 train/test digests (261 below three first groups),
8,565 train/validation (350 below three), and 476 spanning all three splits.
Frequency above three groups alone does not establish harmless background.
Narrative passages and memory documents under different source IDs need explicit
alias review; shared database tables need an explicit background policy. A second,
bounded source pass is preserving complete ambiguous memberships and all groups
for a future overlay. No records, exclusions or admission decisions changed.

The CPU eval timing audit found that client timeouts leave synchronous generation
running for 60–137 seconds, blocking later cases. The source fix adds nonblocking
socket disconnect stopping criteria, peer checks before rendering and generation,
normal handling of closed connections, and bounded request timing/token diagnostics
without prompt content. It stops between generation steps and cannot interrupt an
in-progress forward pass or prefill. Root checked and approved the frozen v7 server
and evaluation plan, starting at step 9000. The gated handoff waits for completed
v6 step 8500 before stopping its watcher/container and preflighting v7. Training and
historical evaluations remain intact. Syntax and pin checks passed; actual
cancellation behavior still needs observation in live evaluation.

### 2026-10-04 04:40 UTC resumed monitoring

Main full Muon training reached step 9000/13165, 72000 exposures, zero skips.
Local GPU 97%, DGX Qwen 96%; DGX available memory 33GiB, swap stable 412MiB.
Both Luna v50 slots have 130/132 completed cases; Bunny v51 has 66/430.
Bunny v52 exact 430-case complement is separately approved and its rollover
controller waits for v51 terminal accounting; there is no third active worker.
DGX development checkout fast-forwarded to 7d9ac37.

Installed exact proposed v7 eval unit and wrote separate root handoff approval.
Boundary controller natlang-lfm25-eval-v7-boundary-handoff-20261004 is active,
waiting for v6 step8500 completion. It stops only v6 evaluation, requires no
active owned work, verifies port free, and preflights separately approved v7.
Handoff controller SHA de568e4ec0212a39893f0766a1aa1d8fcb2e3901ea5e00e611a198e7775af184;
plan SHA 35e589bbe926f3a31f04a25b30fabe69c04b8c7b55f1316587cf67ca218451f7.
If v6 starts9000 before handoff, fail closed and preserve its evidence; replan
future boundary rather than duplicate calls. Main training stays active.

S1 source worksheet reconciles 654999 train gold/checked candidates; labels alone
are not final admission. Teacher705705 and failed844 stay separate. Full ambiguous
alias overlay completed: 611 under-three digests are genuinely exactly two-group
reuse, affecting3341 distinct split/record pairs and371 split/group keys. Narrative,
memory and OpenAlex document aliases require heldout-preserving closure proposals.
Database/shared background reuse is reviewed separately; neither high frequency
nor zero source-to-question exact matches proves absence of leakage. No records
or admission decisions changed. Agents continue concrete checker/rejection review.

### 2026-10-04 owner priority: crisp skill self-improvement

Owner requests handover/sync then a transition from general generation to a rich
self-improvement corpus, especially editing crisp SKILL.md assets and reuse on
separate tasks. Focus is now testing/fixing this pipeline and measuring actual
query improvement, then collection at scale. No new general refills should be
approved; drain current reviewed campaigns after skill cases are ready, preserve
all artifacts and free providers for skill authoring. Existing Bunny v52 waiting
controller must be explicitly reconciled before cancelling or redirecting.
Neuralese remaining work is summarized at the top of plans/neuralese/HANDOVER.md.

### 2026-10-04 06:42 UTC Pop self-improvement continuation

Source/plans use frequent small Git commits and ordinary fetch/merge/push of main.
Do not closely track the concurrent DGX developer; continue this work and resolve
conflicts as they arise. Preserve their checkout; Pop builds use isolated worktrees.

At06:16 the DGX had a **global OOM**, killing Pop pilot-v2 and general Qwen v10.
This was not evidence that pilot MemoryMax8G was exhausted. Interrupted artifacts
remain untouched and are not model-quality negatives. Qwen server survived.
At06:38 resumed the exact pinned Qwen v10 launcher with `--resume`, after verifying
launcher SHA dad3fbbbb8247256f1b625e6b166e6f69d63ab670175348e1b9b89ada059c267;
it passed its source/runtime/authority/resource checks and is processing the
655final/205partial partition of860. No new general refill. The optional
preflight-only command refused an existing pool plan, without writes; the actual
resume performed its own gates and exclusive ownership check. RAM available34GiB
with three Pop episode collectors plus resumed general generation.

Own live units: natlang-pop-skill-pilot-v3-20261004 (3 original optimization episodes,
1 worker) and natlang-pop-skill-incorrect-pilot-v1-20261004 (5 objective families,
2 workers). Both use pop-runtime-v3-21b7439, seal
f56e0dc44f4610e37cc1b0bc0a10d5bbf1dd942d9152587f8b51c8197f1d87b1,2 experiments,
2 sealed ablations. Incorrect input SHA
feb25fe4e4cf311fb78a6361353a6048623912b32b2faa6a5502f04a352847ae;60 references audited.
First knapsack support already scored1; its proposed boilerplate was rejected,
then old v3 wasted sealed calls and failed on a binpacking transfer iteration without
withLimit. This is not a positive training example. Luna is diagnosing why that
error escaped as whole-episode failure instead of a scored target failure.

New code391c2de avoids sealed calls for unpromoted/retained baselines; unfinished
searches remain incomplete. fed2a20 clarifies already-bound inputs, immutable
`folder.snapshot()` evaluation and batching independent file writes. New export
bundles reserve a fresh directory exclusively before offline replay (2040152),
fixing the observed concurrent-writer race. Positive replay→materialization→
publication end-to-end fixture is being exercised, not merely unit admission checks.
Actual live improvement rate and positive admission remain outstanding. The packet
audits cover80train/24protected validation episodes; scale only the train partition
when live outcomes and exact replay support doing so. See S2_SKILL_AUTHORING.md.

Local full Muon run9620/13165,zero skips,GPU100%; periodic v8 step9500 execution
eval active. Both Luna general-generation slots remain active; Bunny v51 drained,
v52 waiting refill cancelled. No local Bonsai generation. Dev artifact mirroring
continues separately from Git source synchronization.

### Pop continuation: Git synchronization and replay repairs

Owner clarified that concurrent DGX development should not distract this agent
from self-improvement work. Synchronize source through small commits and
fetch/merge/push of main; handle conflicts normally. Preserve the other checkout's
uncommitted work and keep Pop runtime builds in isolated worktrees.

General Qwen v10 generation is now deliberately paused, superseding the running
status above. Its 655 final / 205 partial cases are retained; exact root-launch.sh
--resume is the continuation route. More than 200 outstanding general requests
were delaying skill pilots. Both Pop skill pilot units remain active; no positive
skill corpus has been admitted yet. First flawed bin-packing skill episode stopped
on an unavailable training evidence reference; inspect actual arguments before
changing evaluator authority. Validation/query outputs remain sealed.

Runtime fix 8c75e42 tracks iteration promises, containing floating rejected
iterations without altering awaited errors or Promise.all behavior (34 tests).
Recording fix 6560804 detaches model exchange snapshots from mutable caller
messages and transport objects (10 tests). Helper 0640045 records effective
runtime turns independently of raw wire retries (3 tests); collector/export
integration is still pending. Raw tool-call parsing alone cannot reproduce
synthesized text tool calls, aliases, retry token totals or truncation exactly.

Exact author invocation capture and positive author-to-replay-to-publication
regression are in progress. Trace diagnostic previews do not provide exact child
inputs; never infer those inputs from final selected source. Replay also exposed
unstable skill event timestamps/invocation IDs in model-facing training feedback.
Future source-side feedback will omit those two telemetry fields, retaining raw
host events and all skill revision/path/action/order information. Do not normalize
old recorded requests to hide this mismatch. Freeze a new runtime after fixes pass.
Preserve failed/interrupted artifacts; exclude them from positive SFT and do not
invent DPO pairs.

Latest local full Muon checkpoint: step 9780/13165, 78240 trained examples,
zero skips, GPU 100%. Main training remains uninterrupted.

### Replay fixes verified; fresh live collection prepared

5facef5 completes opt-in exact host-only rewrite input/output capture (8MB cap),
collector immutable effective-turn recording with raw wire attempts in separate
logs, strict effective-turn schema checks, and exact child replay. bb6b220 projects
only volatile event IDs/timestamps from model-facing training feedback and bumps
source-evaluation to /20. Whole and child replay remain strict on older artifacts.
Root rebuilt and ran the combined model/record/export/positive E2E suite: 26/26
passed. The positive scripted regression yields two support-only rows and a staged
publication proposal, with no provider calls or query/ablation SFT leakage. This
is a regression fixture, not newly generated training data.

Next live packet is Pop v5: 15 train episodes, 240 cases, 150 groups, 60 independently
recomputed optimization bounds; provider-free audit passed. Packet input SHA
47f254e51cfd7563726770add2fb78bbde84d5888072ad44a0377c85438a7a82.
It selects five first incorrect optimization families and both selection designs
for active-license, active-urgency, authorized-access, completed-delivery and
exact-multi-edit, by declared category before query outcomes. No protected
validation episodes included. Settings: three workers, four experiments, two
sealed ablations. Source is built from Git commit5facef5 in isolated DGX worktree
/home/werg/natlang-remote/pop-skill-build-v5-20261004; runtime freezing and launch
are pending. Legacy pilots remain diagnostic evidence; first incorrect knapsack
support search promoted a candidate, but sealed result has not completed yet.

### Fresh Pop v5 is running; legacy pilots retired

DGX service natlang-pop-skill-pilot-v5-20261004 is active. Runtime seal
751f4eaf1cb077fded117ca078035438a1f8ed120014ed0fce40bf2d264a1261
(20749 files), queue input and settings match the preceding 15-case packet.
Runtime/output directories are pop-runtime-v5-5facef5 and pop-pilot-v5 beneath
/mnt/external/natlang-development-data/runs/dgx-development-generated/
crisp-skill-self-improvement-20261004. Node22.22.0, MemoryHigh3G/MemoryMax6G,
no blanket episode timeout. Initial service RSS/cgroup usage about442MB.

Stopped both legacy Pop units through systemd after v5 launched. Their known
recording/redaction gaps prevent positive admission, so continue on v5 rather than
spend more calls producing incompatible legacy recordings. Partial source search
results and durable exchange logs are preserved. Some interrupted old collectors
exited without result.json and their old queue recorded collector_failed_without_artifact;
these particular operator-stopped attempts are infrastructure interruptions, not
model-quality negatives. No completed positive legacy episode was observed. The
old original bin-packing episode finished evaluated/neutral with baseline retained.
Fresh v5 repeats all five incorrect-skill families and adds ten selection/tuning
episodes. Current training checkpoint9860/13165,78880 examples,zero skips.

Next: review v5 results, classify failures from exact traces, replay any actual
positive offline, stage and activate only proven support SFT after source-policy
review; then expand the 80 train episodes. Protected24 validation stay excluded.
General Qwen remains paused; both Luna general slots and local training remain
active. Source Git changes are pushed; artifact mirroring remains separate.

V5 model requests began about7m24s after launch; three workers wrote effective/1
records and raw wire logs. One early weighted-tardiness case retained the baseline
with “No supported hypothesis”; the new not-promoted guard skipped sealed query/
transfer calls correctly. No actual positive admitted yet. Future collector
verification7b62f93 streams every file hash at concurrency2 with per1,000-file
progress logs; c4071e2 reuses two64KiB buffers instead of allocating per file.
Integrity regression passed, including missing/tampered files. Speedup is not yet
benchmarked. Live v5 runtime and queue remain pinned, untouched by these changes.

### 2026-10-04 — New adversarial self-play work

See [ADVERSARIAL_SELF_PLAY.md](ADVERSARIAL_SELF_PLAY.md) for design, code surfaces, current exercises and outstanding work. User wants ChessT plus uncommon semantic games and word games, improving reusable skill bodies and discovery metadata through adversarial experience. Arena, canonical ChessT adapter, three original semantic engines, private policy execution, skill-authoring hooks and offline replay integration are committed. Real Qwen exercises are underway; no positive self-play training admission claimed yet. Preserve live v5 frozen runtime. Source synchronization remains small Git commits/pulls/pushes; generated data uses the existing mirror. Do not overwrite the other DGX developer's checkout.

Adversarial follow-up: two word families are integrated; 11 role episodes are now queued on DGX service `natlang-pop-adversarial-v2-20261004` using frozen `runtime-v2-bd1b590` (details and seal in ADVERSARIAL_SELF_PLAY.md). 38 focused checks pass. Both live word matches, Clue Intercept, Evidence Bluff and a short ChessT tactic replayed exactly. Negotiation exposed an absent-message `undefined` field; fixed and covered by persisted replay test. The prepared v1 runtime is unlaunched and must remain unused. All collection outputs are unpublished until exact replay and admission review. No measured positive self-play skill gain yet. Existing local full training was at step 10,340/13,165 with zero skips; DGX artifact mirror remains active.

### 2026-10-04 — Efficiency landscape expansion

See SELF_IMPROVEMENT_SOURCE_CATALOG.md and training/self_improvement_tasks.json. Built 90 exact optimization episodes (1,080 cases, six new objective types / ten application families), 54 Python code-golf episodes (324 tasks, 18 held-out program templates, 5,184 real sandbox reference checks passed), and 200 pinned MIT KernelBench source tasks (159 train/41 validation, 161 structural groups). Artifacts live at runs/self-improvement-expansion-20261004 and are copied to the DGX development generated-data root. They are task candidates, not admitted SFT. CUDA tasks are explicitly held for numerical contracts, sandboxed compilation, fresh correctness checks and isolated same-device timing; never score throughput under concurrent generation/training contention. Further executor backlog is explicit in the registry.

Scorer identities pin new objective modules plus the code sandbox. Code-golf checks gate byte credit, finite JSON is required, resource interruptions stay unscored, Docker containers are explicitly cleaned up. Existing original-five builder defaults stay scoped to their supported instance formats. Exact replay exposed elapsed_ms/seq/version leaking into authored skill feedback: source-evaluation/21 excludes telemetry there while preserving raw host records. Positive export replay now passes; old frozen runtimes remain untouched. Merged remote decision/compaction work through Git; no other developer checkout overwritten.

Efficiency launch: DGX service natlang-pop-efficiency-v1-20261004 is active, two workers / three experiments / two ablations / one reviewed attempt. It selects replica0 empty and metadata designs for all sixteen new application families (32 episodes) before model outcomes. Source isolated worktree pop-efficiency-build-v1-20261004 is commit7d87127; frozen runtime-v1-7d87127 seal c8ec16995f0cf7b5f9cfc21781d1272432e90a8f42ae4c4430a6c94c7eeee319 (20,785 files). Queue/runtimes under runs/dgx-development-generated/self-improvement-expansion-20261004; registered prepared inputs also mirrored at runs/self-improvement-expansion-20261004 for development. DGX ARM64 sandbox real test passed; all ten focused builder/sandbox checks passed there. Local regression suite32/32 passed including positive staged publication; broader objective/graded suite27 passed before two integration fixes, which have now passed focused rerun. Pinned KernelBench source cache is also copied to external vendor/datasets/kernelbench-20261004.

Recent live incompletions: one v5 editor passed a mutable folder instead of snapshot; two adversarial searches produced invalid iteration state or exhausted editing turns. Preserve exact traces and diagnose rather than mark as task-quality negatives. Further source prompt clarifies that allowedFiles is a permission list, not a requirement to create every optional file; this post-launch improvement is not in runtime-v1 and should enter the next fresh snapshot. Elapsed telemetry replay fix is in runtime-v1. Local training step10,620/13,165, zero skips; original v5 and adversarial queues remain active. No measured positive from the expansion is claimed yet.

Post-launch audit correction: efficiency-v1 was safely operator-stopped after about2m with only its first two exact-objective tasks started; no code-golf task executed. The original batched Python runner exposed other test inputs through introspection of runner locals (gold outputs were still host-only). Independent Luna audit is replacing this with per-input isolation before a fresh v2 runtime/queue. Preserve v1 attempts as operator interruptions, not model-quality negatives. Registry now has pinned collection references and reports terminal/unfinished/positive-candidate counts separately from prepared tasks and admission. Further state errors now identify the invalid field path and reject sparse arrays instead of silently turning holes into null; seven folder regressions and two preparation/inventory tests passed.

Resolved expansion isolation hold: commits7719b90/125ee03 isolate every Python test in a fresh subprocess and protect batch-parent memory with Linux PR_SET_DUMPABLE=0 (fail closed). Frame-walking and same-UID parent-memory regressions pass; observable allocation exhaustion is unscored, SIGKILL cause remains explicitly unproven. Independent Luna audit also found ASCII/Unicode separator contract mismatch;46efb05 creates code-golf-v2 with exact ASCII-space/tab/LF rules plus mandatory CR/form-feed/vertical-tab/NBSP/NEL cases, normalized-source-byte metric documented. Legacy golf-v1 is retained and explicitly excluded in registry. v2 full audit324 tasks/5,184 checks passed under isolated evaluator, no provider calls. Combined local objective/builder/sandbox/positive-export regression24/24 passed; updated boundary builder4/4 and DGX combined11/11 passed.

Fresh service natlang-pop-efficiency-v2-20261004 launched with two workers, three experiments, two ablations, max-attempts1, no blanket episode deadline. Same predeclared 32-family/design slate, new golf contracts; input SHA c75069504875a8107773edd5896fdb921e7a8ce35065a9e6f276bcdf346b64ee. Source Git46efb05, isolated build worktree unchanged; frozen runtime-v2-46efb05 seal c98d313843649d88d674d169fae744abbc23e6495e04f65d538f51be88f873e4 (20,786 files). Both v1 operator interruptions preserved; zero old code-golf model calls. Expansion active totals144 episodes /1,404 audited problem instances plus200 GPU-executor-pending source tasks; do not count excluded golf-v1 again. KernelBench and all task/audit data mirrored to DGX development tree. Further priority is review actual improvement rate, then implement cache/compression/SQL-work/IR/numerical/GPU evaluators from registry backlog. No expansion SFT admission claimed.

### 2026-10-04 — Additive semantic tasks and research sources

User added contextual time series, hard puzzles/CSP/crosswords, programming/natural/low-resource/author-language translation and beefy semantic classification requiring research. Source catalog now reviews SciFact (explicit claims CC-BY-4.0 / abstract ODC-By-1.0), ContractNLI (CC-BY-4.0), Evidence Inference, FEVEROUS, LegalBench, SciRIFF and AVeriTeC (also potential static conversion of annotated research QA, not fabricated tool trajectories). Original controlled fictional research service packets seed the executable path; do not call them a beefy external benchmark. Natural translation remains held for reliable semantic alternatives; “author languages” clarification pending.

Prepared: 18 operational time-series episodes/72 case appearances, 54 bounded JS/Python-to-new-expression-language episodes/162 cases (7,938 independent public-source behavior checks), 15 CSP episodes/168 appearances/56 unique puzzles (12 train/3 protected nonogram validation). CSP v1 integration audit caught expected solution instead of required bound; preserve old packet, collect only corrected csp-v2. Crossword initial packet has only one support group and is held pending true second support puzzle. Research-v1 had one single-document reference; research-v2 adds the necessary sensor evidence (4 episodes/16 cases/38 verified reference spans), full two-document checks now pass. Author views omit raw service implementation code; runtime exposes signatures and callable services. Pin every custom scorer in collection/replay identities.

Artifacts under runs/self-improvement-expansion-20261004, registry training/self_improvement_tasks.json; task preparation is separate from SFT/DPO admission. SciFact source archive acquired at vendor/datasets/scifact-20261004, SHA11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be; source adapter in progress. Further work: substantive external adapters, valid alternative-evidence handling and connected-document splits; exercise fresh frozen semantic queue and review improvement rate; escalate puzzle difficulty after seed executor validates. Existing live frozen v5/adversarial/efficiency-v2 untouched and still active, efficiency requests continue making progress. Local full training last step10,860/13,165 zero skips. DGX main has another developer's dirty files; synchronize source via Git, never overwrite them. After task completion resume requested interruptible50-minute monitoring cadence.

Semantic follow-up: crossword-v2 now has two genuinely distinct support puzzles per family and one query (9 episodes/27 cases), hash7108877d9cc5f884cd5f8903887114448fc7860c191887f1ea9adae491e1977d; 27 reference checks passed. SciFact adapter a0a2ad1 prepares809 train candidates (332 support/173 contradict/304 insufficient-evidence), 490 connected claim/document groups; labels and evidence host-only, default no dev/test. Packet scifact-source-v6 is preparation only, needs retrieval/distractors/evidence-set executor; earlier v1–v5 are preserved intermediate candidates and excluded from registry totals. AVeriTeC currently links CC-BY-NC-4.0 and stays held from general training pending use-policy review. Registry totals244 prepared episodes/1,849 audited case appearances plus1,009 executor-pending external source tasks (200 kernels+809 SciFact); unique problem count differs from repeated skill conditions. No new SFT admission claimed.

Fresh semantic queue launched on DGX service natlang-pop-semantic-v1-20261004:23 predeclared first-train episodes (one/family),2workers/3experiments/2ablations/1attempt, no blanket episode deadline. Protected nonogram validation excluded; SciFact source candidates require executor before joining. Isolated build worktree pop-semantic-build-v1-20261004 Git4a104e3; runtime-semantic-v1-4a104e3 seal54eef6e3afc7e0c079fe3d593ff18183c05061f6a966dde7e8e461b13eebc523/20,808files. DGX focused14/14 tests passed before freezing. Input/output in DGX development generated-data self-improvement-expansion-20261004/semantic-pilot-v1 and queue-semantic-v1; inputSHA in its manifest and task registry. All prepared artifacts including SciFact adapter candidates mirrored externally. Other developer's dirty main preserved; Git source synchronization via fetched commit and isolated worktree. Review exact traces/replay any actual positives before admission; controlled seeds do not yet demonstrate robust skill gains.

### 2026-10-04 — Expansion following actual failure review

SciFact now has a bounded executable evidence scorer and 108 episodes covering all809 source cases once (647support/162query, globally disjoint connected document groups). Packet `scifact-episodes-v2/episodes.jsonl` SHA2b52698e4a91c0beba7e1c439bb8031d77afc47424617f6164671474a5b95b37; 809 independent scorer/reference checks and structural/target-load gate passed. Scope is supplied cited abstracts, not open-web truth. Earlier unbounded/missing-metric v1 is superseded and retained. Registry counts raw SciFact preparation as backing source, avoiding double counting. Active preparation totals352episodes/2658case appearances, plus200 GPU-executor-pending kernel tasks. Collection/admission remain separate.

NusaX preparation has25,200 tasks from600 parallel rows in42 directions, with all translations of a row sharing one role; original400-row test stays protected. Dataset CC-BY-SA-4.0 license is preserved separately from Apache code. Static SFT is held for source/IR/upstream-rights review; self-improvement also needs a semantic evaluator accepting legitimate alternatives. Do not exact-string grade translation. Research-landscape-v1 (54episodes/432appearances) is explicitly held: some mandatory evidence is filler that does not determine the verdict. A substantive v2 needs independently derived decisions and causal evidence-necessity checks before collection. Neither held corpus contributes to active totals.

Actual expansion outcome audit is in `audit-actual-v1`: twelve completed results, eight not-promoted/four incomplete, zero admitted improvements and zero DPO pairs. Eight retained perfect support baselines. Two edited candidates had invalid skill metadata; a model score claim rejected by the host consistency guard is an author/control failure, not evidence that the host oracle is broken. Preserve failed edits and exact measured feedback for later matched recovery examples. Current source-evaluation/22 validates skill YAML before evaluation; the live older runtime is unchanged. Fresh support-only headroom screening should exclude saturated baselines, retain potentially repairable zero-quality starts, and never select on query outcomes.

Fresh support-only screening is active on DGX `natlang-pop-semantic-headroom-v2-20261004`, four concurrent episodes, all112 train research-v3/SciFact episodes, band0..0.95 and no family skipping. Input SHA59cefbd2fd349573ecc796e85279c5935de71ddd8714b52cdc0ad485f973420f, folder `semantic-headroom-v2` under DGX development generated expansion root. Frozen runtime-semantic-v2-66ad8d8d seal43be75fc25b644242350584753e82b42bc3625d27134364bda31aa2fb9844e51 /20815files includes source-evaluation/22, label aliases and scorer pins; five actual ARM focused checks passed. No query/transfer is screened. New `run_screened_skill_queue.py` waits for the finite screen, independently joins every row and retained episode to the pinned original packet, checks runtime/executor identity and exact retained query content, then launches collection; empty selections end explicitly, partial/changed packets fail closed. It admits nothing. Three handoff fixtures pass. Existing older queues remain pinned and their incomplete attempts retained. Latest local full training11320/13165, zero skips; both GPUs active.

Source-evaluation/23 follow-up prevents known fixture/resource-timeout baseline placeholders from becoming paired quality gains; diagnostic caching and cancellation remain intact. New headroom code excludes these unscored screens; exporter rejects old artifact pairs containing them as well. This fixes a latent path, not an observed admitted false positive. All809 active SciFact cases also explicitly checked to have nonempty positive evidence alternatives; new scorer rejects malformed future empty alternatives.16 focused regressions pass. NusaX candidate Program/2 adapter e834bd4 now exists, still held due conflicting upstream SmSA license declarations/rights evidence; no natural-translation SFT admitted. ContractNLI and substantive fictional research-v2 source work ongoing in owned agent files.

ContractNLI v4-r3 is now prepared:697 focused episodes/5151 unique document-hypothesis decisions from303 original train contracts. All5151 scorer/reference checks, global role/target-load audit and4 builder/scorer fixtures pass; source projection review independently matches official annotations. Packet SHA0a5da21218ff1a9e35dd5375abba972b6d22c7268e01925bdd62f59923218cae.41 all-hypothesis bundles(v3) and v4/v4-r2 drafts are superseded/retained, not counted twice.120 train contracts/2040 decisions held because normalized long source paragraphs connect to original dev/test;61dev/123test remain protected. Paragraph closure does not prove paraphrase/article separation. Code/scorer pins are integrated for future frozen collection. Source revisioneced6528dd3c1d14d73f9a87df8f7bdbc03126f9, archive SHAe03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b, dataset CC-BY-4.0. Active preparation totals1049episodes/7809case appearances plus200 pending GPU tasks; neither holds nor admission are silently counted as generated training data.

Screen batches: source-evaluation/22 pending whole-sweep collection waiter was stopped before any collector launch. Fresh source-evaluation/23 runtime-semantic-v3-dfe17451 seal38d74a7e289d56411b8003b7a82344200531ccc62aca8cdc21a0944427aeb1ef (20819files),13 actual ARM evaluator/export tests pass. Fixed input-order8-case batches of the112episode screen can collect as each batch completes; immutable source/query bytes and selected screen rows are independently verified. Shared whole-collector pool limits these new queues to4 collectors total to avoid RAM oversubscription (DGX had22GiB available with13 inference requests running). Pool waits are interruptible resource allocation, not task deadlines. Existing live v5/adversarial/efficiency/semantic-v1 queues remain pinned. Batch plan is under semantic-headroom-v2/batch-plan; new queue launch names/receipts must be checked, not inferred from prepared inputs.


### 2026-10-04 — Screened expansion deployed; first positive under review

Fourteen fixed input-order batch waiters `natlang-pop-semantic-v3-b01-20261004` through `b14` are active on DGX. They share four collector slots (including ContractNLI); they are not 28 independently concurrent collectors. Isolated source worktree `pop-semantic-build-v3-20261004` stays fixed at d66b3ce7. Collection runtime v3 source-evaluation/23 seal38d74a7e289d56411b8003b7a82344200531ccc62aca8cdc21a0944427aeb1ef. At the latest observation, batches01/02 have selected5/4 episodes and zero terminal outcomes; later batches still wait for complete support screens. Prepared batch plan/receipts are under `semantic-headroom-v2/batch-plan` and `semantic-headroom-v2/batches`; outputs `queue-semantic-v3/batch-NN`. Never infer completion or admission from an active service.

ContractNLI pilot has17 first-input-order episodes, one per hypothesis, inputSHA8d204cb5e279652fc26f0430989eb0f93a0ac1842e648dd4b14a581d906b3e0b. `natlang-pop-contractnli-headroom-v1-20261004` screens support only; `natlang-pop-contractnli-collection-v1-20261004` waits for the complete screen. Files are in `contractnli-pilot-v1`, not `contractnli-headroom-v1`; output `queue-contractnli-v1`. Isolated source `pop-semantic-build-v4-20261004` stays fixed d66b3ce7; runtime-semantic-v4-d66b3ce7 seal7ead2903bfc0749a51705c2991143d73a7d5365e3b123f9b76777abcd94bffa2.5 actual ARM focused checks passed before freezing. Generic ContractNLI structural audit now also recognizes its manifest:697 episodes/5151cases, zero errors and warnings.

Older semantic-v1 queue now has9 terminal/23, with its first positive candidate `timeseries-delayed-intervention-empty`. Luna is checking exact offline replay, held-out outcome/visible-input integrity and source-bound export; NOT admitted yet. Preserve failure/success context, do not claim a DPO pair without matching effective prompts/options. Local full training at11580/13165, no skip problem observed; generated-data mirror remains active.

Controlled fictional research v2/v3 remain held. Root manual review found that v3's resolver requires active suppliers on every route while visible prose scopes that condition to the standard route, and reconciliation-found mutations do not change rendered source text. Four construction tests passing did not establish text/oracle consistency. Repair visible global conditions and missing-record rendering, test each actual changed source passage, then review again before collecting. NLLB full source expansion/static human-reference pipeline integration is ongoing; arbitrary small sample caps should not hide available rows. NusaX remains held for upstream rights conflict. Avoid large translation rebuilds during local RAM pressure; preserve completed artifacts and bounded work.


Time-series visible-contract follow-up: future builder now emits explicit allowedCategories, and delayed-intervention prose names `model-consistent` and `deviation`. New immutable timeseries-v2-visible-labels packet18episodes/72cases SHA2f1e95cc5e9db74ee073b88a8d0675995ef8fc9939fd75eb852a2b0545e221ef;3 fixtures and structural audit pass. The older live apparent positive still needs effective-visible-input audit before any admission. Never rewrite its pinned packet in place. Registry update is with the NLLB adapter agent; collection remains unlaunched for v2.


Positive audit correction: delayed-intervention-v1 baseline performed the arithmetic/temporal reasoning correctly but emitted `consistent`/`inconsistent`; hidden exact gold used `model-consistent`/`deviation`. The selected skill corrected only this omitted vocabulary. Treat this as task-contract repair, NOT genuine self-improvement; held admission regardless of exact replay. Exact replay used zero providers, exporter quarantined all turns because one of16 decisions repeated an earlier call (15 independently approved). Future exporter may retain independent approved decisions with complete original contexts/action linkage and explicit rejected-decision metadata, but no gate was relaxed here. Current expansion still has no admitted genuine positive. Use revised visible-vocabulary tasks; consider a weaker separately deployed executor after local full training rather than spending many teacher calls on saturated baselines.

Full NLLB preparation75bba4c:241528 rows/39actual directions,153262support/39960query/48306validation,6559exact-text components and2blank-row exclusions in physical row ledger. Full-v2 preserved with exact source disposition; earlier capped packet superseded. Source-bound static-reference SFT builder/checker/native admission implementation ongoing, held until root review; no self-improvement translation reward yet. Corpus inventory must stream these ~727MB IR files, not parse all rows into local RAM while training.


Future queued collection now accepts separate --executor-endpoint/--executor-model, preserving the author endpoint/model. Queue identity pins overrides and forwards them to the already-supported collector; screened handoff requires that executor to match the original support screen.11 offline queue/handoff fixtures pass. No live queue is changed or student server deployed. This enables using the trained weaker student as executor later while retaining Qwen as skill author; inspect actual student task failures before selecting families.


Research seed review complete: 118105f6/ebdb18cb explicitly render reconciliation found flags, global registration conditions and unresolved workflow prerequisites. Fresh `research-landscape-v3-consistency-candidate` packetSHAc96973053b2ec796afc73cc42fa73ed66f5492806560b85b363e9b930e28b7c3 passes4 fixtures, structural audit and19 semantic reference checks for4cases. Registered as one prepared episode, NOT collected/admitted or a broad benchmark. Earlier v2 and v3/prosefix drafts remain explicit held/superseded entries. Active preparation now1050episodes/7813case appearances plus200 GPU-executor-pending tasks; full natural-translation candidates remain separate held static-SFT sources.


User skill-ergonomics follow-up: new source-evaluation/24 makes each skill and supporting-file read target explicit, explains choosing relevant skills and applying their procedure, keeps soft metadata and on-demand bodies, and clarifies authors should start with one useful SKILL.md. Missing validation traces are deliberate; page train evidence one case at a time rather than request hidden diagnostics.33 focused checks pass. Collector/screen pins now include disclosure and skill parser; fresh deployment required. Existing source23 queues remain pinned. Superseded capped NLLB sample state corrected to excluded_superseded_full_source_candidate so it cannot inflate pending-source totals.


Presentation deployment: DGX isolated `pop-semantic-build-v5-20261004` fixed c37fac7e;25 actual ARM skill/soft-metadata/headroom checks pass, fresh runtime-semantic-v5-c37fac7e source24 seal a2575ed1f89780bf70af61677c45edc6b831f8f35cf83f1963899ece7fcb1571/20858files. Service natlang-pop-semantic-presentation-v1-20261004 launched,2workers/3experiments/2ablations/1attempt, same shared4collector slots. Queue `queue-semantic-presentation-v1`, input `semantic-presentation-v1/episodes.jsonl` SHA55fe24a17ffd35b26a3b56eca19569e1530c8adc99ff9838a2b673615e0caaad. Selected first3SciFact cases002/003/004 from the first complete support-screen batch, without query selection. This is another retained authoring attempt on the existing tasks, not3new unique task preparations. Does not change old frozen screens/queues. Wait for exact replay and genuinely transferable gain review; no improvement claim from a launched service.


Generic Python metric isolation audit (runs/generation-check-20261004/python-tests-oracle-isolation-audit): candidate and private test module share a Python process/mount, so readable fixtures can be inspected, while restrictive inherited umask can instead make the harness unreadable. Some sandbox failures also become quality0.92 scanned current expansion/adversarial episode/manifest files show no python-tests use; current code-golf uses the isolated code-objective runner. Future episode scorer now explicitly holds python-tests collection pending a separated oracle and unscored infrastructure failures; it does not pretend the arbitrary test harness is private. Direct legacy grading utility remains unchanged for diagnostics, not an admissible collection metric. Basic objective/graded scorer construction also requires its code pin. No current active corpus loss identified.

NLLB review:617 support-source cases yield617 approved final answers and617 held prefix/context turns; never claim1234 training targets. Root independently checks full bundle substitutions after recomputing artifact checksums, including answer/input/source identity/hidden-input and removed policy. Missing NLLB policy fails explicitly instead of silently falling into default direct-answer holds. Downstream tokenization/visible-input audit is with Luna; full153262support bundle expansion awaits that review.


### 2026-10-04 14:44 UTC DGX resource rebalance and recovery in progress

Qwen server is healthy after reducing gpu-memory-utilization0.65→0.45, keeping
max-model-len65536, max-num-seqs256, batched-tokens8192 and all model/parser/backend
settings. Actual engine allocation80996→59356MiB (about21GiB freed), new KV cache
34.74GiB/2997409tokens/~45.74 full-context requests. Eight active requests observed
after resumption; capacity256 is not a measured optimal concurrency. Other current
Neuralese jobs retained; no obsolete training containers were running. Local full
Muon's approved context limit is16384, matching the617-row NLLB downstream audit;
that static lane still needs explicit combined mix admission, not a full unweighted
153262-row append. TypeScript check passed.

Maintenance had an infrastructure incident: clients were frozen and20 requests
drained, but the helper's10-minute cold-start limit rolled back before model loading
finished and thawed clients before restored server readiness. Several support screens
then recorded transport errors rapidly; semantic-v2 completed112 with75SciFact error
rows. These are infrastructure evidence, not model-quality failures or exhausted
targets. All original attempts/screen rows remain. Warm retry at0.45 is now healthy;
all remaining units thawed (absent transient units need separate recovery). Luna
quality agent is preparing exact error-only support recovery packets/commands using
frozen source-evaluation24 runtime, new screen/output identities, preserving band
exclusions and sealed query bytes; root must review/launch them. Do not rerun an
original queue over unfinished directories, nor count maintenance errors as negatives.

Cold HDD prefetch took451s; warm prefetch1.24s, but per-expert host-to-device weight
copies took582.8s. A180s bounded pause of competing GPU clients/training did not
establish a material speed gain; those jobs resumed. Cache reclamation during
loading increased free memory but can cause another cold read of active weight
inputs. The hygiene service now requires Qwen health before cleanup; actual cold
condition skipped and healthy-path validation is recorded. Post-load targeted
reclamation freed~17GiB clean cache, without file deletion/global cache flush.
Receipts and exact commands: runs/generation-check-20261004/dgx-resource-rebalance-v1
and DGX ~/natlang-remote/resource-rebalance-20261004. Old65% container retained
stopped under natlang-qwen36-nvfp4-server-reserved65-20261004 for rollback.

Separate observed failure: natlang-label-clef-flash-20261004 was OOM-killed at its
8GiB host MemoryMax (MemoryPeak exactly8GiB), despite26GiB combined ledger claim
and --memory-gb24 CUDA cap. It was not restarted. Its host startup allowance needs
review and renewed ledger admission; script preserves matching teacher/case manifest
and appends/skips existing IDs. A freeze request timed out while the process was
blocked; the unit was explicitly thawed afterward. No global OOM claim is supported.


### 2026-10-04 15:15 UTC DGX screen recovery launched; cache scopes separated

Exact original-order recovery inputs75SciFact+12ContractNLI independently matched
original episodes and unscored transport rows;42 successfully scored rows excluded
from retry, regardless of score. New frozenruntime-semantic-v6-0d19c05e seal
fa22a6960d50941c29442f06df48932931d2c2843ec751f8fc25471c4e0e86bf;
24actualARM skills/softmetadata/author/headroom/blind-view/time-budget/selection
checks passed, and it includes origin's outage-ready support screen wrapper.
Two active screens use actual *recovery-transport-v2 input directories; v3 folders
are unlaunched proposals only. Semantic input591c03e9...04516 concurrency8;
ContractNLI4da2ceb1...2f4859 concurrency2. Band0,.95/probe0/support-only retained.
No new unique prepared tasks or training admission inferred.

Launched12 batch waiters: natlang-pop-semantic-recovery-b01..b10-20261004
and natlang-pop-contractnli-recovery-b01..b02-20261004. Fixed input-order chunks8
(last3/4),2workers/3experiments/2ablations/1attempt, shared existing4whole-collector
slots. Output queue-semantic-recovery-v1 / queue-contractnli-recovery-v1,
batch-NN; kept files in actualinputdir/batches-live-v1/batch-NN. Inventory registry
reflects actual command paths, not agent's unlaunched proposal paths; input/batch/
runtime joins pass full inventory with0errors and1050/7813prep totals unchanged.
First screen completions/positive quality remain unclaimed. Transport incident
audit and root operational report under generation-check-20261004/
dgx-resource-rebalance-v1; launch receipts in each live input directory.

Refined cache hygiene: completed dataset/corpus/export cleanup runs independently
of Qwen availability; only the model-weight invocation uses --ready-url /health.
Four offline tests pass. Actual unready endpoint skipped all weights with durable
model_not_ready receipt; deployed oneshot follows both scopes. This supersedes
the first whole-service ExecCondition so offline Qwen cannot block unrelated
clean-cache reclamation. Readiness is a resource guard, not a language deadline.
Observed Qwen13–15requests,~4%KV,0preemptions,30-second shared-workload sample
~48generatedtokens/sec; not a controlled throughput comparison or optimality proof.
Further host-cap/streaming checkpoint hash review delegated to Luna for Clef OOM;
no additional model was restarted yet. Local full Muon latest12060/13165,0skips.


### 2026-10-04 — Bounded label-checkpoint hashing

Luna committed e6b32cf: label_decision_cases.py streams checkpoint hashes in1MiB
chunks rather than reading an entire shard. Two provider-free tests prove exact
old/new manifest identity, including sorted paths/empty input file. Clef source
has four3.9–5.0GB safetensors shards; the previous hashing step could temporarily
allocate almost5GB. This fixes a real avoidable allocation, not a proven diagnosis
of the earlier8GiB host-limit OOM. No model restart/deployed-source mutation by
this audit; the other DGX process has independently queued a replacement label
job through memory_ledger (PID484110, host-max26GiB); actual GPU launch was
not observed. Synchronize code through Git and retain existing matching label manifest.

### 2026-10-04 visual/frontend/3D task research

User requested expansion into HTML/CSS/frontend JS, SVG and 3D. See
[VISUAL_FRONTEND_SELF_IMPROVEMENT.md](VISUAL_FRONTEND_SELF_IMPROVEMENT.md).
Eight families are registered as executor-pending in the existing central task
registry, not prepared/admitted episodes. Prioritize project-generated responsive
layout repair and semantic SVG diagrams, then JS state-machine repair and
requirement-checked parametric CAD. CADTestBench and cadgenbench are promising
verifier/method references; DesignBench includes code-based edit/repair modes.
External source licenses/lineage still require pinned review before import.
Existing host executeCase/scoring hooks fit artifact evaluation, but browser/CAD
isolation and verifiers remain to implement. End-to-end image conditioning/replay/
training has not been established, so start with text-visible requirements and
numerical render feedback. Skill body and description tuning should be evaluated
with held-out family transfer and ablations. No running queues were changed.

### 2026-10-04 visual/frontend source intake integrated

Implemented pinned acquisition, source adapters, blind-input/provenance auditing
and registry/inventory integration; see the integrated section of
[VISUAL_FRONTEND_SELF_IMPROVEMENT.md](VISUAL_FRONTEND_SELF_IMPROVEMENT.md).
734 artifact task packets / 534 source groups across nine lanes are prepared in
`data/self-improvement/visual-frontend/intake-v2`, with raw sources under
`vendor/datasets/visual-frontend`. Both are mirrored to the DGX development data.
51 WebSight packets have only the artifact-executor blocker; 683 carry additional
rights/assets/oracle/image/framework holds. These packets are not native
SkillEpisodes, are unassigned to training/evaluation partitions, and add zero
active collector targets or admitted trajectories. Central inventory has zero
errors; existing native counts remain 1,050 episodes / 7,813 problem instances.
New CLI: `.venv/bin/python scripts/run_visual_source_intake.py --acquire`.
Independent browser/SVG/CAD executors and global source-group split review remain
next. Do not feed packets directly into SFT or mount whole raw source roots for a
model. Preserve superseded intake-v1 and all held rows; v2 removes provisional
train labels and binds held dispositions to hashes. CADGenBench targets are
private; CADTestBench declares MIT but reuses unlicensed CADPrompt material,
held pending upstream rights review. No genuine CAD iteration logs were found.

Visual intake correction before publication: registered intake is now **v3**,
superseding the preceding v2 entry. Counts stay 734 packets/534 source-group IDs;
zero admission/collector launch. The review found Angular template filenames were
incompatible with templateUrl; v3 uses source-compatible new.component.* names
and validates relative references. All 16 Angular rows additionally reference a
CSS file absent from their source record; that dependency is explicitly held,
not a model-quality failure. Retain v1/v2 drafts and their audits; the registry's
artifact_source_history retains nine prior v2 entries, and the data policy marks
both earlier versions superseded. This does not migrate or overwrite any running
worker state. Source/CAD/reference/hold evidence and the final intake are mirrored.

DGX source mirror verification completed from isolated Git worktree
`/home/werg/natlang-remote/visual-source-intake-20261004` at c0741517. The full task
inventory, including raw source checksums and packet/oracle/asset hashes, reports
zero errors and the same 734/534 intake totals. See tracked
`runs/generation-check-20261004/visual-source-intake/dgx-verification.json`.
The other developer's main checkout was fetched, not overwritten or merged.

## 2026-10-04: diverse student projection and collector corrections

The corrected native-template base/final GPU packet completed: base 1/23 and
full-SFT student 11/23 successful executions, plus one held source in each arm.
This is a small diagnostic packet, not whole-system accuracy. The original
published LFM template token IDs match the training encoder exactly; training
used one BOS. The previous duplicate BOS was a serving defect.

Projection v2 completed four customer-service reducer cases, with three native
candidate trajectories / ten turns; none automatically entered training. Its
mostly constant-false labels are insufficient for the planned post-training mix.
A broader train-only reference bank now contains 128 current-admitted cases
across 39 families, selected from the pinned historical native trajectory bank.
All 128 are present in the completed parent's training membership and their
pinned prepared-teacher source groups map explicitly to train. Full protected
identity alias exclusion remains mandatory. Qasper's three pending-equivalence
references were held. Selection reports preceding that hold contain 131/40;
use the reviewed IR for current counts. No evaluation answers are reused.

Course changes / probability contract:
- Teacher initialization now executes its actions against fresh current contexts
  and current runtime contracts, then checks the entire outcome. Old system
  prompts/tool descriptions need not match. Retired actions are not silently
  aliased. MH prefix replay still requires exact request/observation equality.
- LFM's 65,536-wide output head contains IDs absent from its 64,402-entry
  tokenizer. Projection generation masks these padded IDs, and both target and
  guided likelihoods use the identical restricted/renormalized support. The
  scoring identity is now `closed-assistant-tokenizer-support-temperature-logprob/2`.
  This is a conditional distribution over decodable IDs, not the old unrestricted
  output-head distribution. Old receipts are never rescored or mixed in-chain.
- Canonical initialization excludes the native template's formatting newline
  after assistant EOS: actual generation stops at EOS. EOS itself remains scored.

V3 and v4 rounds were safely interrupted with receipts preserved after these
collector defects appeared. Their stopped-review receipts explain the changes;
no training publication. V5 is running on the local GPU:
`natlang-student-projection-mh-v5-diverse-20261004.service`,
`runs/student-posttraining-20261004/projection-v5-native-terminator`, plan SHA
`5079d52f240e015381ca3d3225cbfce09981ab323192fcb22cfa10100a95ca6e`.
It uses the immutable completed full-SFT adapter, eight suffix proposals per
case, fresh execution and native admission. First teacher replay admitted.
Seven targeted JS/Python checks passed, including asymmetric MH accounting,
strict prefix replay, padded-logit support, and canonical EOS formatting.

Next: review v5 yield/failures and family/label diversity; admit and register
verified candidates through the existing lineage/token/mix/source gates; prepare
a modest rehearsal-plus-projection phase with unchanged protected held-out rows;
then launch the resumable Muon phase via its reviewed manifest. No live
post-training phase has started. DGX self-improvement/Neuralese jobs remain
running and were not restarted; observed GPU utilization was 93%.

### 2026-10-04 projection failure review: agent guidance and replay identity

V5 reached 26 completed cases / 13 candidates / 87 native turns at the first
50-minute check; zero teacher replay failures. Render preview of v2's ten
candidate turns retained nine (616 supervised tokens); one repeated call was
explicitly unapproved and filtered. These are previews, not corpus publication.

Failed suffixes expose concrete ergonomics gaps: models read source handles
after moveTo, choose destination handles as the source, repeat identical failed
code, fabricate properties such as board.state, and sometimes convert blocked
plans into success after our rejection feedback suggests doing so. New folder
prompt guidance shows source-to-destination movement, destination verification
and unchanged handle paths. return_result feedback now repairs blocked/failed
reason/value fields and allows success only when the task is actually complete.
Existing frozen v5 remains unchanged; review these prompts in a later immutable
round, not by editing its runtime. Strict answer/file/honest-stop checks remain.

One live-inventory suffix had a prefix-observation mismatch. native/values.ts
uses a process-global liveId counter, also printed in agent livePreview. This
is a plausible fresh-replay nondeterminism source and remains to investigate
with an exact failing-request diff and task-scoped identity design. Do not
normalize away arbitrary observation differences or admit that rejected move.
Recovery training should emphasize API inspection, reuse of completed judgments,
checking state after failed writes, and honest stops; do not train raw failed
outputs as positive SFT or form DPO pairs without shared-context verification.

### 2026-10-04 live-identity replay defect confirmed and fixed

The train-only live-inventory case was replayed twice with identical teacher
actions and seeds. Both old runs passed the outcome oracle, but two of three
requests differed. With execution-local display identities, both outcomes pass
and all three requests are byte-identical. Evidence:
`runs/student-posttraining-20261004/live-identity-replay-review-v1.json`.

NativeRuntime now owns a display identity registry (shared by native invocations
within one public runtime task). Agent argument/scope previews, eval observations,
staged values and stored-local diagnostics use it recursively. Aliases retain
one ID within the execution. Process-global serialized trace identities remain
unchanged. Replay request equality remains strict; rejected historical moves
are preserved, not reclassified or migrated. Build passed, 83 targeted tests
passed, and an additional native-session fresh-replay regression passed.
Sealed runtime v42 (`runtime-v42-x64-student-recovery-r1`, manifest
`7bfff6cad6b9b161427a71e35c56481c2dc946d4a20dd9e45e79bba92c81affe`)
contains the fix and the preceding stop/file guidance. V5's frozen runtime is
untouched. Next collect a small targeted recovery round after v5 releases the
local GPU; compare execution yield without claiming broad model improvement.

Recovery round v6 is queued behind successful v5 completion and final-summary
publication; it holds no GPU resources while waiting. Unit:
`natlang-student-projection-mh-v6-recovery-r2-20261004.service`. Plan under
`runs/student-posttraining-20261004/projection-v6-recovery-guidance`, SHA
`623d932b771ff95c6889e60a4ea91750a2bc37506477d46bfa453976056dea90`.
Eight exact train-only references: two route-planning, two folder-criteria, two
live-inventory, one event-retry, one contract-diagnosis. Fixed student adapter
and search controls match v5; prompts/runtime differ deliberately. Source closure
is a pinned exact subset of the reviewed broad packet; preflight passed eight
cases with zero provider calls. Collector code is copied and pinned within this
round to keep later repository edits from invalidating its queued launch.
First planning attempt used a wrong family label and produced no plan/provider
calls; its failed transient unit is retained, r2 is the actual queued launcher.
V5 reached 42/128 complete, 21 candidate episodes / 156 turns; candidate counts
remain separate from published training data. 84 targeted tests passed in total.

### 2026-10-04 task-skill discovery and reliability work

User wants failure clusters converted into generally discoverable task skills,
then skill-use SFT if needed, and a defensible supported scope reaching 90%
ordinary end-to-end success. Projection candidate yield (~50%) is NOT that
accuracy. Qualification must use frozen rules based on visible task features,
fresh source groups, coverage/cost reporting and uncertainty. No "too hard"
labels or 90% certificate have been created. See STUDENT_RELIABILITY.md.

Implemented streaming failure audit (multi-label, review required, no gold/SFT/DPO
creation) and six task-oriented skills in training/student-skills/task-workflows-v1.
Names: calculate-from-data, move-and-verify-files, inspect-callable-apis,
judge-against-criteria, complete-the-call, resume-after-partial-effects. Each
description gives an observable use condition. Only the last is specifically
recovery; the others apply before failure during normal execution. The initial
recovery-named draft was superseded before provider calls and archived in its
planning packet. Description tuning must measure helpful/missed/unnecessary
skill selection, execution benefit and cost, not merely number of skill reads.

A collector integration gap was fixed: direct Program IR roots previously used
raw definitionNode and never bound companion skills. Public calls and collector
root execution now share prepareDefinitionNode; collector keeps its separate
oracle/effects/replay orchestration. Root discovery and read_code disclosure
passed a native execution test, along with 23 related tests. Per-function child
binding inside direct collection remains an explicit follow-up audit.

New ordinary-execution runner evaluate-student-skills.mjs and sealed runtime v43
(07cdec07252b0e61ae68a5abf1e408352d4a7a6f82f735c658748237e3015a7e)
are prepared. Eight-case baseline/discovery/instructed development ablation is
queued AFTER v6 succeeds: natlang-student-skill-ablation-v2-20261004.service.
Plan SHA 90ed3c9d628f256396e87a08eb11ddaa0388db5a7c3ae4809842adae62ae7ab1
at runs/student-posttraining-20261004/skill-ablation-v2-discovery. Preflight
passed eight cases/three arms/zero provider calls. Library, runner, server,
student, IR, source closure and runtime are pinned. No teacher hints; no automatic
training admission. Its small train-only development packet cannot certify
reliability. Review results before deciding whether to collect skill-use SFT,
change descriptions/procedures or expand to qualification. V5/v6 remain unchanged.

### 2026-10-04 skill-first scheduling and shared guided student decoding

Pulled/merged origin/main guided-generation work (merge 848e839b). Added an HF
adapter around the same Python Guide, with explicit --guidance-module, optional
request guidance, forced required-tool prefix, token bans and cache reconstruction
on rollback. LFM hybrid caches are not snapshotted. Output allowance, retry count,
and 3x generated-token attempt allowance bound work; receipts retain rejections,
accepted-after-exhaustion points, discarded tokens and prefill-token work. No code
executes in the Guide: "run" means repeated character chunks. Actual execution
and task admission remain the final checks. Guidance + natlang_projection is
explicitly rejected because backtracking proposal probabilities are not accounted
for in our MH sampler. No current MH search distribution was changed.

Fixed a shared Guide false positive: function-looking text inside quoted code
arguments is no longer classified as an unknown transport tool. Python tests:
32 passed (student rollback/bans/budget/Unicode plus shared Guide and response
parser). Corresponding llama.cpp guidance implementation/conformance needs this
quoted-name fix as a follow-up; do not claim all backends already match it.
Guided image pinned locally at
sha256:345578df62932b8ab247c7a726c63c951b0b9d3df9a8b6d1a950e27292e41eba,
base eabc88d83cba..., tree-sitter 0.25.2/typescript 0.23.2. Live guided student
results are pending; tests alone do not establish any execution improvement.

User prioritizes iterating general skills/discovery and teaching correct usage
with SFT where prompted use helps; search-based on-policy post-training follows.
Reprioritized local GPU: paused broad v5 with 56 complete cases, 29 candidate
episodes / 207 turns (NOT published training). Existing receipts unchanged.
Interrupted proposal must not resume; v7 is a fresh remaining-72-case plan.
Course receipt: runs/student-posttraining-20261004/skill-priority-course-change-v1.json.

Current order:
- Ordinary skill v2 RUNNING under natlang-student-skill-ablation-v2-priority-r2-20261004.
  Exact original plan/runner/runtime/weights remain unchanged; only launch order
  changed. First priority launch hit the previous container's port while it was
  stopping, identity check rejected before task calls. Log retained as
  startup-port-collision-v1.log; launcher now checks adapter identity at readiness.
- Guided companion v3 queued: natlang-student-skill-ablation-v3-guided-20261004.
  Plan SHA 19f0b492186fb57efa4f0dda6dffd91c84987ab347078b24010282be8888766c.
  guided_baseline/discovery/instructed, same eight cases, runtime v43, six skills,
  fixed completed-SFT weights and output allowance. Different bounded decoder
  intervention explicit. Shared quoted-name fix included before any calls; old
  preflight plan archived. All results development-only, not qualification/SFT.
- Targeted MH v6 queued after v3: natlang-student-projection-mh-v6-after-skills-20261004.
  Its reviewed collection plan/inputs remain identical; launcher dependency changed.
- Broad remaining MH v7 queued after v6: natlang-student-projection-mh-v7-remainder-r2-20261004.
  Plan SHA 4382c60869766e293eb6088502b1a4cd1f326ffb18670125eb8d4557c3d41de5,
  under runs/student-posttraining-20261004/projection-v7-remainder. Same v41
  frozen runtime/search controls, 72 selected references excluding completed v5.
  Collector/helper/server copied/pinned. Removed extraneous mutable repo dist pins;
  complete sealed-runtime manifest still verified. Exact teacher subset matches.
  Preflight passed72/0calls; initial plan mistakes were caught before provider calls.

Next: review ordinary vs instructed vs guided skill usage, inspect helpful/missed/
unnecessary reads and failures, refine general names/descriptions/bodies, create
freshly executed/admitted skill-use SFT examples if prompting helps. Do not auto
train all guided/rejected traces. Audit child/per-function binding, collect source-
group-disjoint reliability qualification, admit diverse search candidates and start
resumable Muon post-training phase (not yet launched). DGX GPU observed96% busy;
other agent's jobs/repo untouched. Sync code via Git and data/runtime mirrors.

Skill v2 completed: baseline1/8, discovery0/8, instructed0/8; skill reads in2/8
cases for both skill arms, zero infrastructure errors. Repeated skill reads did
not yield progress. This packet does not yet demonstrate a prompt-induced skill
benefit; don't claim retrieval SFT is sufficient. Guided v3 is now executing.
Its first arithmetic failure repeats expression text inside return_result's
non-code argument, outside the current Guide's eval-code repetition checks.
Consider extending literal-argument/envelope checks rather than treating syntax
checks as a complete solution. Keep original bounded experiment receipts.

Added build-skill-use-demonstrations.mjs: explicit pinned train-only source closure,
shared frozen runtime/library, one relevant skill retrieval, fresh teacher action
replay, complete task/native/curriculum admission. Outputs remain candidates,
not automatic training publication and NOT student on-policy data. Pilot:
runs/student-posttraining-20261004/skill-use-demonstrations-v1, plan SHA
 e60c433ad72e56009dc049007409f38ed95d1d7b4bb340133922ce9b6af9bd5a.
8attempted/7admitted/48turns/0provider calls. Rejected contract teacher uses legacy
read_function not offered by the current runtime. Preserve rejection; do not teach
an unavailable tool or silently admit this old action. Other candidates need the
normal render/token/source checks and review for unnecessary reads before SFT.
A fresh native regression verified that direct collector child calls bind their
own companion skills through the existing shared kernel path (2program-skill tests
passed); no child binding patch was necessary. Code and data/runtime43 mirrored to
DGX clean worktree/development-data; other agent's dirty development repo untouched.

### 2026-10-05 first 50-minute check, skill refinement and phase preparation

Actual50-minute sleep completed; guided v3 and targeted MH v6 finished successfully.
Guided baseline/discovery/instructed all0/8; discovery read skills5/8 and instructed
6/8. No infrastructure errors. 263guided requests;20unknown-tool rejections,
no accepted-after-exhaustion points or attempt-budget hits. Uptake increased but
execution did not. Semantic/API/state/finish failures remain; guided form alone
is insufficient. V6 yielded5candidates/36turns; no probability error. V7 is running,
last33/72complete,12candidates/44turns. Existing frozen experiments unchanged.

Refined task-workflows-v2 (v1 retained): inspect-callable-apis distinguishes function
declarations from invocations and requires using inspected APIs; calculate-from-data
puts expressions in eval rather than literal tool arguments; complete-the-call
returns computed values rather than strings of code. Shared skill listing asks for
an action using what was read, another skill for a distinct need, and reuse of
visible instructions. It permits rereading changed or unavailable instructions.
20skill/native tests passed. Sealed runtime44 manifest
fda8002ca8c1594cbd8c9d1da30b78d1d6b6f2286aedbf56fe6f04d05522b368.

Expanded fresh teacher demonstration pilot:55attempted/49admitted/263raw turns,
28selected families, at most3perfamily; no provider calls. Packet:
runs/student-posttraining-20261004/skill-use-demonstrations-v2-broad, plan
ade4bf26d63300535d5d137ea7c83ea3012fff5e67c70156347b13fbd87956f8.
Relevant skill selected by reviewed task shape; no causal helpfulness claim.
3rejections use unavailable read_function calls. 3reach the right oracle value but
miss required observed evidence: stale page IDs fail, or decisive text was not
observed in fresh replay (teacher code itself repeats the hidden text). Correct
exclusions, not admission conditions to relax. Keep these traces out of SFT.

Render/token preview posttraining-render-v1 combines completed v5, v6 and the broad
skill demonstrations:361usable decisions,24,095supervised tokens;231skill and130
search decisions. Native removed failed/redundant decisions; final token audit
removed13duplicate rendered pairs. Fresh native attestation49rows/244approved
pre-dedup decisions/231ready decisions; exact rendered/native lineage verified.
Not yet a training publication. New catalog classifications keep candidate and
evaluation lanes visible; stopped/invalidated projection rounds require disposition
review before selection. Evaluation traces are diagnostic-only.

Preparing skill-only Muon post-training phase at
runs/student-posttraining-20261005/skill-sft-phase-v1. Explicit selection231skill
rows +128deterministic parent replay anchors (64reducer/64other), all5410protected
held rows preserved, all9prior exclusions preserved, no MH rows in this phase.
Full parent corpus retained; downsampling is a declared phase exposure choice,
not retirement. Copied complete optimizer/scheduler/RNG/weights checkpoint exactly.
Data/token/mix/source/inventory/phase gates and queued training launch still pending
at this entry. Goal: teach skill selection/use, then a separate search SFT phase
and paired ordinary execution eval. No claim of skill benefit yet.

Inventory now accepts an explicitly hash-pinned policy manifest plus phase reason.
Use exact completed-parent policy during this phase, keeping the trainer's strict
curriculum-policy identity check intact; global catalog bookkeeping can evolve.
Tests reject a changed pin (5inventory/phase tests passed). This does not approve
any rows or weaken native/source/token gates. Current catalog policy differs only
in bookkeeping decisions; parent exclusions/admission policy remain fixed.

### 2026-10-05 skill phase approved and evaluation chain queued

Skill-only phase gates passed, including verification against the real copied
parent checkpoint. 359train decisions (231skill demonstrations +128replay anchors),
5410held decisions identical to parent, all9prior exclusions preserved. Reducers
139/359 (38.7%). Ready SHA721922b51de5a87806e66562d3c2489bfb779edc7b129cfecd4f0e982d029572.
Three epochs/1077additional examples, Muon optimizer/RNG retained, explicit new
cosine horizon at2e-5. Targets step13299/trained_examples106388; periodic heldout
loss every50steps and required final heldout evaluation. Exact training plan SHA
 a4b5a839927a8d1eaddfb196bb8d9749e599713d6f18b2937a8f74000b35b033.
Frozen scripts plus PYTHONPATH pin prevent mixed live/frozen trainer imports.

Queued units, each waits for predecessor success and validates expected outputs:
- natlang-lfm25-skill-sft-v1-20261005: after v7 search, runs existing resumable
  supervisor; root runs/student-posttraining-20261005/skill-sft-phase-v1.
- natlang-skill-sft-execution-eval-v1-20261005: verifies plan/artifacts/full checkpoint
  completion, copies immutable weights, same protected packet/runtime/single-BOS
  template and budgets as final-gpu-eval-v3-single-bos. Reuses prior full-SFT
  ordinary result11/23; no duplicate untrained-base model run. Root
  runs/student-posttraining-20261005/skill-sft-execution-eval-v1.
- natlang-skill-use-paired-eval-v1-20261005: after protected report, compares parent
  and skill-SFT checkpoint on8train development cases, baseline/discovery/instructed,
  identical refined v2 library + runtime44. Exact plans materialized/pinned after
  checkpoint snapshot. No teacher hints, qualification claim, or training publication.
  Root runs/student-posttraining-20261005/skill-use-paired-eval-v1.

Latest search41/72complete; no failure requiring restart. DGX GPU92%busy and
self-improvement worker queue active. Preserve other agent's development checkout.
Next: inspect actual phase launch/checkpoints and eval results, then build separately
admitted search-SFT phase using completed MH rows; do not mix guided rollback into
MH p/q scoring. Still need defensible visible-feature reliability scope with separate
held-out qualification. No demonstrated skill benefit yet.

Ported quoted-tool-name masking to private llama.cpp-neuralese fork. CPU actual
nz_guide checks passed quoted Math.max, true forbidden tail with exact rollback
position, escapes, incomplete quoted strings, and rewind. New cross-server fixtures
added (full server conformance not run locally). Fork origin/neuralese1807e9288
includes concurrent f459288ca zero-export port preservation fix. Do not claim active
DGX server or wasm binary includes this change until rebuilt/verified. No server
restart performed. Course choice: maintain shared Python/C++ guidance parity,
without altering task oracles or frozen student experiments.
W&B phase reporting also queued: natlang-skill-training-wandb-20261005, separate
run lfm25skills20261005. Waits for checkpoint identity to match this phase before
reporting metrics. Aggregate metadata/loss only; no task data or weights uploaded.
Paired evaluation template preflight passed8cases/3arms/0modelcalls; training-plan
artifact and Docker pins verified again. Oct5 data and runtime44 mirrored to DGX.

### 2026-10-05 second actual 50-minute check: training started cleanly

V7 completed72/72,39candidate episodes/143raw turns. Skill phase launched without
restart, live step13220/trained_examples105759 at last observation, GPU100%busy.
Checkpoint records preserved Muon/RNG and new phase scheduler. W&B reporting active.
Periodic fixed128row heldout loss0.781331 at13164,0.786388 at13200; no task accuracy
inference from loss. Parent full heldout0.57536 is a different population/aggregation.

Fixed queued evaluator's mistaken `status` versus actual `state` lifecycle key,
and removed dependency on Result of completed transient systemd units (they can
be garbage-collected). Completion now comes from durable supervisor/plan/checkpoint
receipts. Added reusable supervisor `handoff` command: verifies artifact/image pins,
`state:complete`, exact plan identity, full resumability files, target and required
final evaluation, emits state/weight hashes. Regression passes; frozen trainer
unchanged. Evaluation launcher uses a hash-pinned copy of this helper.

Search preview runs/student-posttraining-20261005/search-sft-render-v1:73episodes
from completed v5/v6/v7,244admitted decisions,17474supervised tokens; no pair duplicates
or token rejections. Interrupted v5 case remains excluded. Native/source review still
in progress; this is not an approved training phase and no search SFT launch yet.
V7 failure heuristics:211answer mismatch,75repetition,53state,49API,39incomplete,
30effects,10finish,10truncation,9replay-prefix mismatch (multilabel; not accuracy).
All9prefix mismatches are two cases (TextWorld iterate and workflow security dir).
Fresh no-model replay reproduced TextWorld displayed function-ID shift (#2 versus#1),
while workflow initial prefix reproduced exactly. Keep strict raw context equality;
do not normalize away observations/probabilities or admit rejected moves. Need
follow-up on preview ID allocation and better mismatch diagnostic receipts.
Search preview attestation uncovered a real pipeline lineage bug: nativeRowDigest
hashed undefined-valued in-memory fields, then JSON serialization omitted them.
All244search decisions therefore carried stale trajectory digests. Fixed hashing
the serialized JSON representation (parsed stored rows retain their existing valid
hashes),23native-materializer tests passed. Native refresh from73saved admitted rows
changes ONLY source_ref.source_row_sha256 and teacher_trajectory_digest; verified
all244contexts/actions/targets/outcomes/admission decisions identical. Explicit
lineage-repair.json records original pins. Original files preserved; repaired
render/token output in search-sft-render-v1/repaired. All73source identities checked
against exact reviewed bank, authoritative train groups, original parent train split
and full protected alias union (8.45GB teacher snapshot hash verified). No protected
source overlap. Final native attestation rerun against repaired data. This repairs
provenance, does not convert rejected actions into training targets. Future collectors
use corrected digest once their runtime is rebuilt; already running frozen processes
must receive explicit repair or new snapshot, not silent mutation.
Additional checkpoint edge-case fixed for future trainers: a phase/append transition
must clear inherited heldout_after, otherwise an interruption at the last optimizer
step could mistake the parent's validation metric for a new final evaluation.
Trainer now clears it and records heldout_after_step on final evaluation. New plans
should set completion.final_evaluation_step_required:true. Two focused supervisor
regressions pass. Active frozen trainer stays unchanged; queued evaluator explicitly
requires its final held-out log with exact current examples/loss before handoff.
Correction to earlier wording: this trainer's final heldout_loss is the legacy first100
held rows, not all5410. The periodic128row token-weighted metric is separate. Protected
ordinary execution evaluation remains the task-performance check; do not compare
those loss aggregates as though populations or averaging were identical.

### 2026-10-05 third actual 50-minute check: skill SFT done, search SFT running

Skill phase completed at13299/106388, first attempt, clean exit and full checkpoint.
Final legacy-first100 heldout loss0.58131159. Protected ordinary eval11/23, exactly
same case successes/failures as full SFT11/23; no infra/resource errors. Snapshot
b3232b620d180c1a4f46ec111069684038b430527e559fb8a4f5161f5d82f517.
Paired8train-case skill ablation with runtime44/v2skills:
parent baseline1/8, discovery0/8(read1/8), instructed0/8(read1/8).
skill-SFT baseline0/8, discovery0/8(read8/8), instructed0/8(read8/8).
Retrieval improved but application did not. Discovery90reads, instructed94reads,
mostly repeated inspect-callable-apis. No benefit claim from skill uptake alone.

Added an unchanged skill-body reread reminder in shared native runtime. Full original
instructions and return value remain available; changed bodies reset count; no read
cap or refusal. Trace includes unchanged_read_count.3program-skill tests passed.
Sealed runtime45 manifest40c698e0fd764e4118036881937ae5b1062c5f1b82d9fc42d9b6e11851e1d321
for subsequent developmental comparisons; prior frozen experiments unchanged.

Search-SFT phase root runs/student-posttraining-20261005/search-sft-phase-v1:
436train =244searched decisions +64deterministic skill replay +128parent anchors;
5410held rows unchanged,9prior exclusions preserved, reducer share139/436=31.88%.
Search rows from prior full-SFT student,8-step MH with teacher-conditioned proposals;
NOT newly collected from skill-SFT weights, NOT guided rollback, NOT convergence.
Metadata lineage repair, native/source/token/mix/inventory gates passed. Three epochs,
1308additional examples, targets13463/107696, positive2e-5 Muon cosine, full optimizer
and RNG preserved. New final_evaluation_step_required gate enabled. Plan SHA
581e1a31d931f3c9f5e5883c699ad4b64192b4132fc4f970b0d6ad952f2bb4ef.
Unit natlang-lfm25-search-sft-v1-20261005 active, resumed cleanly, GPU100%busy.
Protected eval queued natlang-search-sft-execution-eval-v1-20261005 (same packet),
then natlang-search-skill-paired-eval-v1-20261005 compares skill-SFT and searched-SFT
students on8train cases under identical runtime45/reminder/library. Development,
not qualification. W&B separate run lfm25search20261005. Follow up actual results,
refine application training/skills and collect fresh next-round search as justified;
do not assume more retrieval is better. Reliability scope qualification still pending.

### 2026-10-05 skill application and infrastructure review

Search SFT completed cleanly at13463/107696 with step-bound final evaluation;
protected ordinary execution10/23 overall versus parent11/23. Do not promote it
as an accuracy gain (semantic_accuracy excludes incomplete attempts). The paired
8hard training cases remain0/8 for both skill and search SFT in all three arms.
Broader preselected14train-case/8family probe ongoing in skill-graduated-probe-v1:
full SFT baseline4/14(two infrastructure errors), discovery5/14, instructed5/14,
no skill reads; skill SFT all arms4/14, inventory arms read14/14. Development only,
not a qualification certificate. Search arm pending. Investigate baseline infra
before a fair comparison; repeated reads have not produced task improvements.

Training design lacked inventory-present/no-read controls. Added explicit static
inventory-control demonstration mode;49/49 original admitted teachers replayed
with skills exposed and no extra retrieval, no model requests. Candidates at
runs/student-posttraining-20261005/skill-inventory-controls-v1; not yet rendered,
source/token/mix admitted or published. Runtime46 sealed manifest
cad1e74d9cf749b249d997ea7d933bc062f1ab06b9d2cdb01c4087882fdeb1ef.
Changed unchanged-read reminder to emit skill-unchanged-read diagnostic; materializer
retains those calls as context rather than positive targets despite reminder text
changing.27focused runtime/materializer tests pass. No refusal or read cap.

DGX queue-v2 completed32episodes with zero positives. Deeper review found25 socket
failure strings in traces; transport exceptions were absorbed by runtime repair and
reported incomplete, bypassing queue retries. New recorder retains actual thrown
cause-chain codes/HTTP status separately, never infers transport from model text.
Collector overrides absorbed retryable failures with provider_failure, positive:false,
retaining original outcome and all evidence. Queue applies existing bounded exponential
backoff only with explicit typed failure receipts; intentional aborts stay interrupted.
Focused JS and Python regressions pass. Existing artifacts immutable; need fresh sealed
authoring runtime and a reviewed recovery queue, not rewrite old semantic records.
Other DGX agent GPU jobs remain active; do not restart their server blindly. External
volume only45GBfree; avoid duplicate model copies. Skill optional-file/frontmatter
prompt fixes already on main must be included in fresh runtime. Some optimization
cases have graded partial rewards; re-evaluate reward ceilings after infrastructure
recovery rather than attributing every zero-positive attempt to reward design.

Follow-up: 14case probe finished; search SFT baseline5/14, discovery6/14,
instructed5/14. Apparent infrastructure errors are ONLY request-budget exhaustion
(full baseline2, search discovery1/instructed2), not socket failures. Collector now
emits NATLANG_MODEL_REQUEST_BUDGET; new evaluator reports resource_limited separately.
Historical frozen summaries remain intact; do not misreport budget stops as transport.
20collector tests pass. DGX fresh transport-recovery-v3 queue started with8reviewed
historical transport-affected episodes,2workers,3attempts maximum,30s exponential
backoff cap300s.13old episodes matched exact host socket diagnostics; old artifacts
unchanged. New skill-authoring runtime seal
02f1e1508a4c8ec220f8ab31da3dad12045eabfedccd4efc15667abc360c797f
includes typed recorder/collector fixes; uses parent skill-authoring dependencies.
Static49controls materialized195positive decisions, ordinary tokenizer audit retains
186 (9exact duplicates),12743supervised tokens, lengths<=10171, no protected overlap.
Balanced skill phase being prepared from original full-SFT checkpoint, not from
regressed search checkpoint. Controlled development experiment; no benefit assumed.

Balanced skill SFT now running natlang-lfm25-skill-balanced-sft-v1-20261005,
root skill-balanced-sft-phase-v1, from exact completed original full-SFT checkpoint
with full Muon optimizer/RNG.525train after20joint duplicates filtered,5410held
identical; reducer200/525=38.1%. Two epochs1050additional examples, end13296,
target106361, positive2e-5LR; plan SHA
bc15366763b458e723769ea1cd80f8340c75558be9595f643d86199af4afbdad.
Native/source/token/mix/inventory gates passed, parent9exclusions preserved.
Protected ordinary eval queued skill-balanced-execution-eval-v1, then the same
14case/3arm development probe queued skill-balanced-probe-v1. W&B separate run
lfm25skillbalanced20261005. All checkpoints/artifacts retained; no automatic promotion.
Found and fixed readiness receipt reporting global policy hash even when catalog
used explicit pinned phase policy; report itself was correct and validator uses it.

### 2026-10-05 threshold-triggered chunk search and skill topic experiment

User requested implementation/exercise of local chunk rewriting with student NLL
thresholds, teacher-ranked candidates, divergence-triggered teacher continuation and
corrective prefix SFT. Added rewrite-student-chunks.mjs/chunk-search.mjs, parsed native
assistant token-offset scoring in chunk_scoring.py and optional /natlang/score chunk
metadata. Ordinary serving retains lazy loading; no template/EOS change. Factory
create_student_rewrite_pipeline.py now builds ordinary render/token-audit stages for
natlang.student_chunk_rewrite_plan/1. See plans/CHUNK_REWRITE_SFT.md for contract/limits.

Pilot v1 failed staging (student_guidance.py dependency omitted); v2 stopped explicitly
after finding128token teacher allowance generated empty parsed eval arguments. Prior
files retained, unpublished. v3 six cases completed and exercised rewrites/continuations,
but used all context decisions in NLL and had no durable cutoff. Keep v1-v3 held.
Corrected v4 filters NLL/search to native-approved targets; context failures/detours
stay context. Both chunk mean and worst-token thresholds required. Whole-action coarse
fallback handles high trajectory NLL without a fine bad span. Best valid state retained;
changed non-assistant context/tool surface forces teacher continuation, scored one full
action at a time; persistent early stop cannot be swallowed by runtime repair. Student
only proposals supported when teacher absent; changed observations then require prefix
fallback. Proposal score is teacher/student likelihood of its JSON replacement response
in explicit rewrite context, not likelihood under the ordinary task context. All student
eligibility scores use ordinary task context. Teacher ranking includes wrapper/length
bias; thresholds exploratory, not calibrated. All exact oracles preserved.

Native materializer now honors provenance.student_chunk_rewrite.supervision_cutoff_decision
on re-export, retaining suffix as context only; invalid indices fail closed.28focused
JS tests and8Python tests passed. Operator code and source/runtime/weights pinned;
all candidate wire/score receipts retained. Prefix targets use complete admitted action,
never synthetic mid-action EOS. New candidate artifacts registered review-pending in
training/data_sources.json; no automatic phase inclusion. Future phase policies remain
pinned independently from catalog bookkeeping.

v4 root runs/student-posttraining-20261005/chunk-rewrite-pilot-v4; unit
natlang-chunk-rewrite-pilot-v4-20261005, plan SHA
4c197fb6ac6bbdd76e25f11681b9675657fa2f92d3fd6255a70ceca1ec6f424a.
Six reviewed train cases, original full-SFT snapshot, local student + DGX teacher through
natlang-dgx-chunk-teacher-tunnel-20261005(port18082). Runtime47 manifest
d6771580f90f3f75cd22f2e9d73815fdb26e862f8458b83c2defc210d810ec8a.
First corrected case paginated relational query passed fresh task/native checks:
NLL0.6642→0.5257, target tokens444→357, requests4→2; one accepted chunk edit and
one accepted teacher continuation. Still hard, exported one complete corrective prefix
instead of claiming entire trajectory met thresholds. Not downstream accuracy evidence.
Follow full pilot, native/source/render/token audits before any new SFT.

Balanced skill phase ordinary protected score10/23, snapshot
213b1ddf971e3d2a30937d863e6fd56cc52c5f17e1af75749e74a17f10d242b5.
Its14case dev baseline5/14, discovery4/14(read11), instructed3/14(read12), no errors.
Still no reliable skill benefit; original full-SFT11/23 remains better general candidate.
New four-topic candidate evidence-and-contracts-v1 focuses visible evidence/pagination,
exact query scope, claim polarity/group comparisons and callable/return contracts.
Not a replacement; names/topics/library size jointly change. Queued same14train-case
three-arm comparisons for original full-SFT and balanced student in skill-topic-probe-v1
unit natlang-skill-topic-probe-v1-20261005, after v4 ends. Measure task success and
application/cost, not reads alone. Skill SFT uptake did not prove procedure usefulness.

DGX transport-recovery-v3 still active; typed UND_ERR_SOCKET failures recur and bounded
retries are recorded, no longer masked as semantic incompletes. Root cause unresolved.
Do not label whole-episode retry as a connection fix. Local idle-socket reproduction
with/without Connection:close did not reproduce the failure; further diagnostic evidence
needed before claiming keepalive bug. Both machines remain supplied with reviewed work.

## 2026-10-05: preferred training-time repair mode

User strongly prefers selective training-time difficulty detection over mandatory
upfront search. Added opt-in trainer NLL gating from the same autograd forward,
complete corrective-action prefix loss, resumable policy identity, and durable
checkpoint-visible repair outbox. Approved phase manifests can change this explicit
loss policy while preserving optimizer/RNG and existing protected-source gates.
`prepare_online_repair_batch.py` selects latest hard flags against an exact checkpoint
adapter and reviewed executable-source plan; missing sources/deferred programs stay
visible. Existing chunk rewriter handles bounded student-biased teacher regeneration
and fresh native validation. See `plans/ONLINE_REPAIR_SFT.md` for commands and limits.
Do not claim a speedup yet: fused-loss overhead, flag fraction and post-training
accuracy need measurement. Full trajectory truncation needs merged train chains;
single-action rows retain the whole corrective action. Periodic dispatcher and
admitted-repair append loop are not yet running.

Earlier six-case corrected chunk pilot v4 completed: seven positive decisions,
four corrective prefixes, one vanilla row, one accepted teacher-continuation row.
Improved NLL on four programs; these are shaping metrics, not downstream accuracy.
Earlier v1-v3 artifacts remain review-only. UTF16 replacement offsets now have a
non-BMP regression check; frozen v4 predates this fix.

Skill topic probe v2 tests four general procedure topics with baseline/discovery/
instructed/directly-provided arms. Full-SFT results: 4/14, 3/14, 4/14, 5/14 respectively;
discovery read skills in only one case. This tiny development sample suggests a
possible availability-versus-selection distinction, not demonstrated general gain.
Balanced-checkpoint probe also completed; inspect its immutable summary before
choosing skills for another phase. Skill iteration remains an experiment, not an
automatic library promotion.

Balanced checkpoint topic probe: baseline5/14, discovery3/14 (10 reading skills),
instructed2/14 (11 reading), provided2/14 (11 reading). This reinforces that more
skill reads alone do not demonstrate utility; keep original full-SFT as the stronger
candidate. New online-repair tests plus existing trainer loss/phase/append tests ran
in the pinned training image:23 passed,2 skipped (CUDA tests, CPU invocation).

## 2026-10-05 07:00 CEST: online-repair phase active

Started `online-repair-phase-v1` from original full-SFT weights+Muon optimizer/RNG:
409train rows,581source decisions,59multi-action chains,5410protected unchanged,
reducer share26.4%. Three epochs1227examples; global endstep13318/target106538.
PlanSHA42b5c51b790e3ef1f818e575a6fe822735086b95737a01712fbf5aae6fd865a9.
NLL1.2/token8, fullgold every10. First128committed visits:63flagged65unchanged;
recent retained token fraction85%. No accuracy/throughput gain claimed yet.
Unit `natlang-online-repair-training-v2-20261005`, reporter
`natlang-online-repair-wandb-20261005`, W&B `lfm25onlinerepair20261005`.
Initialv1unit failed unsupported CLI `--plan` before any training; corrected positional
invocation. Frozen phase code predates the log-only fix which now reports remaining
corpus epochs rather than dividing global example count by local phase rows.

Merged-chain admission now preserves source/admission fields and source-row hashes,
checks exact token histories at every target and all target termination; merger never
crosses program/split/source-group identity. Generic preparation falls back to originals
with explicit reasons instead of silently losing data. No fallbacks in this phase.
Found three obsolete reference trajectories targeting `read_function`: s1/s5 contract
sound and s5 scoped module pricing. Nine decisions held out explicitly in selection;
not the same nine IDs as parent exclusions, which remain preserved independently.
Need current-runtime regeneration of those sources, not admission relaxation.

One repair round queued to run after phase+finalevaluation completion, using exact
stable checkpoint adapter, up to16current flagged reviewed programs, DGX Qwen teacher,
local student scorer; pinned UTF16-fixed server and operators. Unit
`natlang-online-repair-round-v1-20261005`; configSHA
395ef38d851d09073848cdb1c7e6d3c25b18026dbca7ec9686f20c24659fe2bc.
The worker uses the supervisor's real `complete` status (covered by regression),
then renders/token-audits candidates. After completion inspect `repair-round-v1/result.json`,
fresh native/source lineage and prefix cutoffs; build admitted-repair phase from
this newly trained checkpoint, preserving optimizer/protected rows. Automatic append
and perpetual rounds still need completing. Do not run a second student server
alongside this phase on the8GBGPU. DGX recovery remains active; SocketError provider
failures recur (code-golf filtering ended provider_failure after bounded retries).

## 2026-10-05 07:45 CEST: phase complete, Maple evaluation preparation

Online-repair phase completedstep13318/examples106538, retaining its resumable
checkpoint. Same128-row periodic held loss0.781331 atparent13164 versus0.786851
at13300; no accuracy gain inferred. Final latest per-row flags165/409hard,244unchanged
(about60%unchanged). Repairv1failed before modelcalls because trainer stores adapter
files under `checkpoint/weights/`; snapshot/batch helpers now handle the actual
layout, with regression. Repairv2failed immutable pin check because the snapshot
had accidentally copied/generated Python bytecode; freshv3excludes caches and runs
with bytecode writes disabled. Earlier failed artifactdirs kept intact.
`natlang-online-repair-round-v3-20261005` is now running16programs; v3configSHA
b327a18c6710f0294c933fa77a22b48a834a9fa77ae7d321a88cf5f54399ec51.
Already exercised vanilla reuse, corrective prefixes/teacher continuation, local
code-line repair. Review final admitted/token/source receipts before next training.

User requested evaluation of DeepGrove Maple Preview and explicitly correct inference
stack/template/authenticatedHFdownloads. URL `/maple-preview` currently404; official
HFmodel/GGUF cards and `deepgrove-ai/llama.cpp` fork exist. Evaluation root
`runs/maple-preview-evaluation-20261005`. Official fork pinned
7e30f3adb34b444c3527f94c3343612d71b47d0d, HF GGUF revision
f5466f918e0c50cdb9d4d47a6f35813509a42a30. Primary TQ2_0/F16head file6,349,082,144bytes,
SHA2fad7b4903781aca6b1171881f2d273989fe1735b6b3ea704bc486b472d98039.
Use official embedded Jinja template (supports tools, tool responses, explicit
reasoning prefix), no LFMtemplate. Official runtime instructions target CPU; CPU
build completed onlocalRyzen8645HS with systemGCC. FirstLinuxbrewGCCbuild failed
OpenMP/glibc link mismatch, retained logs; system toolchain fixes it without codechanges.
Download authenticated via savedHFtoken. OldhostHFhub0.29.2 lackedXet; isolated current
HFhub/hf-xet environment now downloading (unit natlang-maple-download-v2-20261005).
Also preparing official fork build + authenticated download onDGX ARMCPU under
`/home/werg/natlang-model-evaluation/`, separate fromdirtydevcheckout. Do not claim
Apple published200+toks/sec applies to ourhardware or agentic task success.
Protected candidate eval runner preflightpassed24cases/23eligible/1policyhold,
original packet/runtime/gold separation and1024output/8turn/16request limits retained.
New genericcandidate evaluator removes LFM-specific checkpoint selection protocol;
pins fixedexternalmodel/stack instead. Needs actual deployment `/props` model path
check, prompt/tool-template smoke tests, model-tokenizer contextfit before execution.
No Maple task results or benchmark measured yet. Downloads/builds are active.

## 2026-10-05 07:53 CEST: Maple integration decision and repair interruption

User requests merging publisher Maple support into our llama.cpp fork if evaluation
shows it useful. Reviewed DGX `/home/werg/llama.cpp-neuralese`, clean branch neuralese
at5d999c0c5. Fetched publisher main into refs/remotes/deepgrove/maple-main, leaving
branch/worktree unchanged. Publisher adds8ce8ca6c6(model support) and7e30f3adb(README)
beyond an ancestor already in our fork. However our fork already has newer upstream
Maple support3d10bcd19. A non-mutating merge-tree probe reports five conflicting files,
including independently added conversion/maple.py andsrc/models/maple.cpp; receipt
`/home/werg/natlang-model-evaluation/maple-merge-preview.txt`. Do not blindly replace
newer upstream code. Compare implementations and officialGGUF loading/logits first;
then merge/port any necessary publisher changes on an integration branch, preserving
neuralese/adapters/cache features. Current upstream requires SwiGLU clamp metadata,
publisher defaults it to7; check officialGGUF compatibility specifically. Publisher
CPU template instructions explicitly require--jinja(thinking prefix included).
Official CPU builds completed onbothlocalx86andDGXARM. Remoteauthenticateddownload
startedunitnatlang-maple-download-20261005; localv2alsoactive,2.4GiBpartial atlastcheck.
The uncommitted genericcandidate runner model identity check now uses actual llama
`/props.model_path`, confirmed against publisher server source; frozen earlierplan
must be replaced with fresh copies/pins beforeexecution. Maple stillhasnoresults.

Repairroundv3stoppedafter7completedprograms because teacher HTTPfetch raised
UND_ERR_SOCKET(other sideclosed) atlocalhost18082. Unitfailed; runner cleanedstudent
container. Preserve outputs and review completedprograms; remainingcases needfresh
attempt with bounded exponential transportretry/durable case progress. This is not
a model quality rejection. No repaircandidate has been automatically admitted.

### Maple upstream provenance confirmed

Upstream PR https://github.com/ggml-org/llama.cpp/pull/27000 explicitly says it ports
DeepGrove commit8ce8ca6c6d with publisher approval. Merged September14,2026 as
3d10bcd19785c7b70626d7ded4a2276ef92bc850, already present in our DGXfork.
Thus no second wholesale publisher merge is currently warranted. Upstream review
added converter fixes and mandatory clamp metadata; check officialGGUF loading and
numerical behavior before deciding on any residual compatibility patch. The saved
implementation diff is DGX natlang-model-evaluation/maple-implementation-diff.txt.

### 2026-10-05 continuation: transport retry and queued Maple benchmark

Repair collector now uses pinned repair-transport.mjs: transient socket/network or
HTTP408/429/500/502/503/504 failures retry atmost3times,5/10/20second backoff with
Retry-After respected. Exact inference payload/seed is reused; no tool side effects
are retried. Auth/request errors and malformed successful JSON fail immediately.
Abort interrupts sleep. Perattempt metadata goes totransport-retries.jsonl without
prompts/credentials. Node regression8/8passed. Freshrepairv4uses remaining9oforiginal
16programs, leaving7completedv3final.json records unchanged; mustjoin/reviewbothsets
before nextphase. Unitnatlang-online-repair-round-v4-20261005, configSHA
fa5053cf7666bdc20ee08f72ed6c4b339abb994d705b41d2ee4dfb5aa6b51315,
collectionplanSHA5f31f5151123d6d198c7dc54497297de024f59b08adb9a87444493ee60bb5623.

Parsed alreadydownloaded GGUF header only(noinference fromincompletepayload): official
GGUF does contain all24maple.swiglu_clamp_exp=7.0entries required byupstream.
Embeddedtemplate exactlymatchesHFchat_template.jinja, SHA
83e4c58ca602ade89b126cc75a036eb8bd06d373d4fd94b09d2277d609131089.
Thus the identified metadata difference currentlydoesnotrequirecompatibilitypatch.
LocalXetdownload was thrashing its2GBcgroup limit(memory.events max129659,noOOM).
Raisedlocalandremote downloadMemoryMax to4GB; localRSSthenfell anddownloadresumed.
Localpartial~5.2GiB atlastreading; DGXdownload~1.3GiB. Authenticatedboth.
Queuedunitnatlang-maple-benchmark-server-v1-20261005 waitsforverifieddownloadreceipt,
thenCPUbenchthreads4/6/12,pp512/tg128,3repetitionsandstarts officialJinjaserver18090,
threads6,context16384,1slot,noGPU. Script/artifacts underMapleevaluationroot.
MemoryMax10GB; inspectactualresident/swapandcontentionbeforereportingspeed.
Newcandidateevalv2planSHA
2e6b9f4c33f7149033348d3556db26b369bdcdc22c55f638796b48b6f0c70d52
preflightpasses23eligiblecases(noinference). Beforetaskcalls nowchecksactualmodel_path,
exactnativetemplatehashanddeployedtokenizercontextfit for everyfirstrequest.
Needhealth,tool/reasoningformatting smoke,thenexecutev2. NoMapleaccuracy/speedresults yet.

Queued Maple follow-through units (all under runs/maple-preview-evaluation-20261005):
- natlang-maple-evaluate-v2-20261005: waitsforhealth, checksnative thinking prefix,
  tool/history rendering, then generates one synthetic tool-call smoke (1024token
  budget). Saves fullsmoke receipt; onlyexecutes protectedv2plan if JSONtool decoding
  passes. Ifsmoke fails, inspectreceipt insteadofrerunningfrozenoutput.
- natlang-maple-dgx-benchmark-v1-20261005: waitsforverifiedlocaldownload, stopsremote
  duplicate download, transferscompleteGGUFvia rsync, verifies remoteSHA, benchmarks
  ARMCPUthreads8/16/20 andcopiesbench receiptsback. Remote partialcache remainsintact.
  These are experiments in separateevaluationfolder, notchanges toDGXuserdevcheckout.
By08:05CEST repairv4hascontinuedthroughadditionalvanilla/prefix/teachercontinuation
cases; no socketretryobservedsofar. Originalmainalreadyintegratedotheragent's
MapleQATplan andmemetic-adapter work. Noactual Maplebench/resultsavailable yet.

## 2026-10-05 09:49 CEST: candidate comparison, sync and repair status

This section supersedes the earlier queued/download-in-progress Maple notes.

- Maple protected execution-v5 is complete: **12/23 eligible cases passed**, eight
  semantic failures, three incomplete; one additional case policy-held. No runtime
  infrastructure failures. Report: `runs/maple-preview-evaluation-20261005/execution-v5/report.json`.
  Primary denominator includes incomplete tasks. This small set does not establish
  a replacement for Ling. Maple publisher-fork v4 grammar failures are infrastructure
  evidence, not capability results: its max-repetition guard rejects valid schema
  char{0,2000}; our newer fork already has the upstream fix.
- Verified Maple GGUF is present on both machines. Embedded native template matches
  the publisher source; runtime `/props` removes only one trailing newline. Source
  and runtime templates have separate pins. Official GGUF has all 24 SwiGLU clamp
  metadata entries required by upstream Maple support. No new C++ change required.
  Explicit `git push origin neuralese` confirmed our llama.cpp fork fully pushed:
  branch neuralese, `5d999c0c5094bedb79f4c675118d4e942f2ac06f`, clean working tree.
- Publisher-build CPU benchmarks (not exact current-fork measurements): local best
  tested decode57.27tok/s (six threads), DGX94.56tok/s (eight threads). DGX20-thread
  result8.37tok/s is anomalous; do not advertise it as scaling behavior. Maple test
  serving now uses our pinned fork on DGXCPU, port18091; local SSH tunnel active.
- Matching **Ling baseline evaluation is running**, unit
  `natlang-ling-evaluate-v1-20261005`, artifacts `runs/ling-maple-comparison-20261005`.
  Original BF16 snapshot9a98e35fe1c9ee255f78dd64771c7ae15a799481, native vLLM
  BailingMoeV3 with ling3 tool/reasoning parsers, native template, thinking enabled.
  Same protected runtime/packet, greedy0,16Kcontext/1024output/8turns/16requests.
  GPU eager native backend; speed comparison must disclose MapleCPU vs LingGPU and
  BF16 vs ternary. The generic evaluator now saves full review-only returned rows
  and hashes, including failures, and duration. Maple v5 unfortunately only retained
  summaries; do not invent missing trajectories. Separate diagnostic reruns needed
  for detailed Maple decision analysis. Shared task execution remains identical;
  only provider identity/tokenizer metadata differ between llama.cpp and vLLM.
- Qwen was idle before this experiment. It is temporarily stopped to free GPU; Ling
  server wrapper restores the existing Qwen container on exit, and local evaluator
  stops Ling in finally. Remote Ling unitv3 has a two-hour resource reservation limit
  as a fallback, not a language deadline. Initial server launch attempts failed due
  to container path/entrypoint mistakes and restored Qwen; v3 corrected those.
- Checkpoint synchronization is complete and verified (23 files), receipt
  `runs/checkpoint-dgx-sync-20261005/verified.json`. DGX canonical development paths
  contain strongest task-evaluated SFT step13164 (11/23) and new online-repair
  step13318. Both preserve Muon optimizer, scheduler, RNG and split state; newer is
  not yet task-evaluated and must not be called best. Original LFM350M base snapshot
  is available offline at `/home/werg/natlang/models/hf/hub/` on DGX. Convenience
  adapter links live in models/trained/. Maple link lives in models/candidates/.
- DeepGrove sibling is **Bonsai500M**, a base Llama ternary model, not an instruction
  tuned small Maple. Verified authenticated download on both machines at
  models/candidates/deepgrove-bonsai-500m, revision5d836f3cbcd11c8ac54d920c7230cda04fbffdac.
  CPU forward/backward through original QLinear passed with finite nonzero gradient;
  receipt `runs/deepgrove-bonsai-500m-20261005/training-smoke-v3.json`. Original custom
  code needs Transformers4.48.3 (isolated target environment; modern5.x import fails).
  It has2048context and no native chat template. Needs a compact curriculum and
  deliberate training template before any SFT run; do not drop it into16K training.
- Repairv3 retained seven completed programs; v4 completed six more, for13/16
  completed,37positive candidate decisions combined. Still held for normal review
  and admission/append. Three remaining references hit turn budget/parallel replay.
  v5 raised repair-only budgets16turns/32requests but was overstrict: matching whole
  historical prompt rejected current prompt/tool-description changes. Diagnostic
  confirms actual task inputs/observations match. New reference matcher retains task
  instructions, tool observations and non-bootstrap actions, ignoring current system
  prompt, tool descriptions and host bootstrap scaffolding. Native full-task oracle
  and source admission remain mandatory. Serial scoring associates parallel calls
  by visible inputs, not completion ordinal. Eight regression checks passed.
- Fresh repairv6 is queued after Ling evaluation: unit
  `natlang-online-repair-round-v6-20261005`, frozen repair-worker-v6/config.json SHA
  dc16f568c88e8a5aa728b81934d5bb4411c54646475888f867e3b2729c3c8691.
  New empty-batch handling records no_admitted_candidates without pretending token
  audit passed. No automatic training publication. Finish reference replay review,
  combine v3/v4/new accepted candidates, evaluate13318 and prepare resumable append
  phase with anchors/protected splits. Perpetual append supervisor remains unfinished.

Cadence: actual50-minute sleep completed08:06–08:56CEST; resumed work after wake.
Continue actual sleep/check-in convention when useful work is waiting; no scheduled
session-wake claim. New DGX user's dev work includes joint KD smoke and Maple BF16
source download; preserve their processes and dirty checkout. Synchronize Git through
clean visual-source-intake worktree, ignored data through existing development sync.

### 2026-10-05 10:01 CEST: Ling provider formatting correction

Ling v1 was stopped after nine case summaries as diagnostic evidence, not a final
capability score. Independent native-template rendering found vLLM's newer message
parser reads only `reasoning`, dropping runtime-v39's `reasoning_content` history.
This removed prior assistant reasoning before the publisher template saw it.
The template itself and tool argument syntax are correct (vLLM parses JSON tool
arguments to objects before rendering; local HF checks must do the same).

Current chat-completion transport now sends both exact reasoning aliases, refuses
conflicting values and preserves the caller's original request. Build completed;
20 focused transport/replay checks passed. For comparison using the unchanged
protected frozen runtime, a pinned local provider-boundary bridge makes this same
conversion and journals exact request/response wires (including streamed usage).
No task instructions, model weights or evaluation oracle change.

Fresh Ling v2 is running/queued under natlang-ling-evaluate-v2-20261005, plan
`runs/ling-maple-comparison-20261005/evaluation-plan-v2.json`; bridge localhost18093
forwards to the existing18092DGX tunnel. Remote server unit now
natlang-ling-native-eval-v4-20261005. All32original BF16 source shards were rehashed
and matched HF blob SHA256,15,787,992,416bytes; weights-verified.json retained.
Independent default-server vs publisher-template proof is queued as
natlang-ling-verify-template-v5-20261005. Check that receipt before declaring the
format fully verified. Failed verification attempts retained as diagnostics.

Repairv6 remains a fresh pinned attempt with sameconfigSHA; launch wrapper now
natlang-online-repair-round-v6-queued-v3-20261005. It waits for Ling v2 to finish,
then for restored Qwen health (up to240seconds), before running the local student
repair worker. Earlier queued wrappers were stopped before collecting any data.
A failed old tunnel unit is a duplicate: port18082 is already served by an existing
SSH process. Do not start a second listener; check actual endpoint health.

Actual-fork Maple benchmark receipt is benchmark-dgx-neuralese-v1.json:
338.90prompt/58.42generationtok/s at8threads, while LingGPU inference was active.
This is slower than the earlier publisher-build figure; investigate shared-memory
contention vs backend changes with subsequent sequential measurements. Do not use
publisher-build speed as the measured speed of our serving fork.

### 2026-10-05 10:08 CEST: Spark unified-memory profiling failure

Remote Ling serverv4 failed before serving any corrected-v2 model request. vLLM
asserted that free GPU memory increased during initialization77.54→94.11GiB;
Spark unified memory/cache reclaim invalidates this profiler's assumption. This
is infrastructure, not a capability result. Qwen restored automatically.
New remote Ling serverv5 uses explicit --kv-cache-memory-bytes2147483648, bypassing
that profiler, retaining same BF16 model/native template/16Kcontext/two slots.
Corrected evaluatorv2 is still waiting for health (no task outputs to mutate).
Additional cleanup unit natlang-ling-restore-qwen-v5-20261005 stops this new server
when evaluatorv2 exits, restoring Qwen before queued repairv6; two-hour reservation
fallback remains. Verify template proof and execution completion on next check.

Added scripts/report_candidate_comparison.py: compares only completed review-only
reports with identical packet/runtime/gold/prompt/tool/budget pins and unique case
identities. Reports all-eligible pass counts, paired wins/losses and task durations;
case time is not raw decode speed. Self-comparison check passes23eligible/12passed.
Unit natlang-maple-ling-comparison-v1-20261005 waits for Lingv2 and writes immutable
runs/ling-maple-comparison-20261005/comparison-v1.json, or fails if no valid final
report. No training publication. Continue periodic rejection/source review and
finish repaircandidate admission/append; no new training run has started here.

### 2026-10-05 10:10 CEST: corrected Ling active; completed repairs salvaged

Explicit2GiBKVcache resolved Ling startup. Corrected evaluationv2 is now executing
cases. Independent native-template verification **passed all24firstrequests**:
`runs/ling-maple-comparison-20261005/native-template-verified-v5.json`. It compares
actual deployed default token IDs against HF tokenizer rendering of the exact
publisher template, retaining reasoning and decoding JSON tool arguments exactly
as vLLM does. Native tool smoke also passed. Keep interruptedv1diagnostic only.

Recovered completed per-program final.json from repairv3/v4 without editing either
attempt. Aggregation `online-repair-phase-v1/repair-completed-review-v1/review.json`
pins13accepted full-task rows,45native turns:37approved decisions/eight context-only
turns. Rendering/token audit running unitnatlang-repair-completed-review-v1-20261005.
No automatic publication; still review ordinary source/admission/split closure and
append through the resumable phase process. This recovers v3's seven completed cases
whose batch did not reach final aggregation before its transport error.

Current units to inspect after next50-minute sleep: Ling evaluationv2, comparisonv1,
Lingrestore-qwenv5, repairround-v6-queued-v3, repaired-completed-reviewv1, development
sync. Observe actual endpoint/container health rather than old failed experiment
units. Qwen restoration is automatic; repair waits for its health before using it.

## 2026-10-05 11:16 CEST: first post-sleep check completed

Actual50-minute sleep completed. Ling thinking-v2 finished11/23eligible passes,
2semantic failures,1file-contract failure,9incomplete,0infrastructure failures.
Maple12/23. Paired7bothpass/5Mapleonly/4Lingonly/7bothfail. Ling114streamresponses
include37length finishes; current transport's truncated-action handling is newer
than the protected frozen runtime. See plans/student-candidate-comparison-20261005.md
and comparison-v1.json. Do not claim the one-case difference chooses a student.

Qwen restoration succeeded. Queued repairv6's initial240second health wait expired
before Qwen was ready, so no collection started then. Restarted the same fresh
attempt after actual health200; initial firstcase now replays12decisions and admits,
with chunk proposals in progress. Future run_online_repair_round now checks teacher
health with a separate900second deployment startup budget before starting student;
five helper regressions passed. No task/language timeout introduced.

Salvaged v3/v4 audit was already finished (not stuck):37usable rows/5132supervised
of202363totaltokens,zero token-audit rejections. Eight context-only turns correctly
filtered. Exact task/source/group identities do not overlap protected comparison.
Still finish ordinary review and new resumable phase (append helper forbids appending
to completed schedules; use posttraining_phase with optimizer/RNG and protected rows).
Current350Mstep13318 has not yet received ordinary task eval, so best-known remains13164.

Queued native Ling thinking-disabled runv3 after current repairworker: unit
natlang-ling-no-thinking-evaluate-v3-20261005. It owns new remote Lingserverv6 plus
bridgev3, uses same explicit2GiBKV and restores Qwen. Planv3preflightpassed23eligible.
No flags/template additions to protected task prompts. Expect results next check.

CPU stack comparison finished in cpu-stack-comparison-v1/ (under Maple evaluation):
publisher30.71/forkauto31.10/forkresident94.61/fork-no-repack86.93decode tok/s. Other
workloads changed across measurements; do not attribute difference solely to loading
mode or change serving defaults without matched followup. Fork can achieve~95tok/s.
DGX other-agent Maple n0 CPU/routing work is active; preserve it. Qwen plus BF16 work
uses~100GiB/121GiB,swap7.6GiB,available21GiB atwake; monitor shared-memory contention.
Stop unused bridgev2/tunnel after completed evaluation; queued next run restarts them.

### 2026-10-05 11:27 CEST: followup launch and remaining-reference recovery

Repairv6 is complete: security directory case added3positive audited decisions,
leavingtwo references held. Combined v3/v4/v6 supply is40positive decisions from14
programs (previous37from13); no automatic training publication. Remaining KQA/S5
mismatches are generated Stored-local summaries and inline-call scaffolding, not
changed original task data. New narrow summary normalization only ignores a local
summary when that whole same variable is printed as complete JSON by the immediately
preceding eval; changed/partial observations still fail. Three matcher checks pass.

Rather than broadly strip remaining observation differences, initial reference
replay now invokes the existing teacher-continuation path at first unmatched context.
It scores actual fresh contexts, retains source/gold separation, and admits only a
complete current native/oracle-valid task. Without a teacher it explicitly holds;
student-difficulty stopping can still hold regenerated references. Changed replay
length is allowed only when a teacher regenerated the suffix; untouched references
still require exact decision count. This is the same continuation principle used
for chunk edits, and does not invent reference matches or relax an oracle.
Fresh worker7 covers only the two remaining targets, queued behind Ling followup:
configSHA9b6f3c35f2246ff10f95532f7427f08597075ea5c674ee9ddd0dda981082ee29,
unitnatlang-online-repair-round-v7-20261005, current900s provider-startup helper.

Thinking-disabled Ling's first queued launcher failed before any model request:
stopped transient tunnel unit had been garbage-collected, so systemctl restart
could not find it. Replacement launcher explicitly creates the tunnel via systemd-run.
Current unitnatlang-ling-no-thinking-evaluate-v3-relaunch-20261005, same frozenplanv3,
remote serverv6; now loading/running. Future orchestration should own/create ephemeral
resources instead of assuming stopped transient units remain restartable. Source
launch scripts and failed logs are retained; no prior evaluation output overwritten.

## 2026-10-05: Maple target selected; repair routing clarified

Owner selects Maple as target student for capability and speed. Ling native
no-thinking followup completed10/23 (4semantic,4contract,5incomplete),485.693s;
thinking11/23,Maple12/23. Small sample does not prove superiority. Long Ling
reasoning is still an open model/serving-behavior issue, not declared fixed by
turning thinking off. Native template/history checks narrow causes but do not
prove all continuation/stop behavior. See student-candidate-comparison-20261005.md.
Qwen container restored; worker7 started preflight and waits actual provider health.

Routing inventory repair-routing-inventory-v1.json:165hard flags,129programs,
97with reviewed sources;16already attempted,81reviewed hard programs remain.
32programs need source-lineage investigation. Narrow batch scope was incorrectly
reported as missing reviewed sources. prepare_online_repair_batch now records
known reviewed programs outside selected scope as outside_batch, separately from
unresolved sources; admission unchanged. Source-scope/helper regression checks pass.
Fresh worker8 selects next16 from109unattempted reviewed source entries (81hard),
queued afterworker7. ConfigSHAfc8f04afb3854d101a84e3ac2443dce940daa0ee11632a355d7259757ab55c8e;
unitnatlang-online-repair-round-v8-20261005. Prior frozen attempts untouched.

Next priorities: finish repair7/8, aggregate/audit the approved40+new decisions,
prepare next resumable posttraining phase without changing protected rows; Maple
M0 representation/parity then M1QAT and exported ordinary evaluation; skills and
self-improvement corpus expansion with actual improvement evidence; neuralese
Maple/joint/nested serving work per its handover. Keep Git and development data
sync running. The350M phase ended at13318; best task-evaluated checkpoint remains
13164 until new ordinary task evaluation. No active350M training at this check.

## 2026-10-05 12:55 CEST: release idle DGX services; local recurrence work

Repair round v9 finished at 12:19. Its CUDA scorer remained resident, alongside an
idle Qwen teacher and completed Maple CPU evaluator. Stopped
`natlang-repair-student-v9-20261005`, `natlang-maple-neuralese-server-v1-20261005`,
and Docker `natlang-qwen36-nvfp4-server`; stopped local v9 scorer tunnel.
DGX available unified memory increased from about 43 GiB to 117 GiB; no GPU
process remained immediately after cleanup. At 12:39:20 the other agent started
`natlang-maple-n2a-smoke`, so do not restart Qwen without admitting its memory
alongside that job. Future repair launches need guaranteed server cleanup.

Local RTX4060 recurrence work uses the actual `train.trajectories` path, including
written child returns reintegrated into caller reads and nested write contexts.
The 350M native model supports 128K positions; 16K is not a language limit.
Port-only measurements reached 16K with batch 2 after attention and prefix fixes;
these are not proof of full recurrence at that length. Actual recurrence first
step passed, subsequent cases exhausted VRAM. Configuring a 32K input admission
cap does not prove actual 32K recurrence fits (largest observed writer context
was 12,958 tokens). The rejected 384-token launcher was never trained and removed.

Activation offload stores saved tensors on CPU without recomputation or detached
write gradients. A prototype storage-address dedup bug corrupted gradients;
`fit-v5-32k/REVIEW-HOLD.json` prohibits promotion or reuse of its checkpoint.
Live-owner/version checks fix that bug; CUDA allocator-reuse gradient regression
and checkpoint/optimizer/memo unit checks passed (7 tests). Fresh fit-v6 uses the
corrected code, Muon, full-state checkpoints, a 32K admission cap, one sampled
written handoff per record, and depth 3. It is a memory diagnostic, not a published
training result, and uses the old v4 port checkpoint rather than the best SFT
checkpoint. No claim of full joint multi-handoff recurrence fitting locally.

Course changes in the recurrence trainer: nesting depth now increments once per
producer edge (previously twice); deterministic producer DAGs share writes and
count each producer boundary once, while cyclic/stochastic/sampled writes do not
share. Muon is optional; checkpoint identity includes input hashes, training
controls and package code. Full optimizer/RNG/cursor state is atomic and resumes
only with matching identity. SIGINT/SIGTERM checkpoint at the optimizer boundary.
Resume and signal behavior still need an end-to-end exercise before a main run.
Repair v8 partial and v9 final decisions still need aggregate admission/token audit.

Local fit-v6 reproduced loss 9.612911 and writer gradient 51.76487 on its first
step (2.77 GiB peak), then OOM in a subsequent producer's feed-forward LoRA
projection despite a 7 GiB activation offload budget. Host swapping also rose.
Do not increase offload blindly on this 14 GiB RAM machine. Need profile exact
producer lengths and reduce transient allocations/graph storage, not filter out
hard cases or detach their contexts. A separate one-record diagnostic exercised
SIGINT checkpoint, automatic reload of Muon/AdamW state, and SIGTERM checkpoint
(`resume-proof-v1`); both shutdowns were clean. This is not a quality run or proof
that the full corpus fits. The copied 7.6 GiB converted corpus is now local at
`data/neuralese/converted-v13-v8`; verify hashes before using it for a main run.

## 2026-10-05: checkpointed local recurrence now fits measured cases

Added opt-in `--ffn-chunk-tokens` (token-local FFN only, full attention/convolution
context retained) and `--checkpoint-layers` (nonreentrant recomputation; complete
cache and recurrence gradients preserved). CPU cached-continuation input/weight
gradient checks and CPU/CUDA chunk-output/gradient tests pass. `resolve_base` now
honors HF_HOME/HF_HUB_CACHE; previously the offline fixture silently skipped even
though the pinned model was available in the configured cache.

Named `--write-curriculum sampled-chain` selects one handoff per nested level,
resampling with checkpointed RNG. Default batch is one, so each selected chain
gets a complete backward and optimizer step; explicit batch accumulates chains.
Other handoffs are crisp context. `joint` keeps all handoffs by default. This is
an explicit training curriculum, not detached recurrence or a runtime limit.
Depth controls selected nested writes. DAG sharing excludes the producer's own
output from dependency edges, matching actual write traversal.

Without checkpointing, FFN chunking reduced first-step peak from 2.77 to 2.51 GiB
but the next producer still OOM at 16,631 tokens. With checkpointing, **fit-v8-32k**
(sampled chains, eight steps) completed in 34 s, peak **5.72 GiB**, largest writer
context **16,631 tokens**, no CPU activation offload; full Muon state checkpointed.
**fit-v9-joint-32k** (joint, same eight input readers) completed in 40 s at the same
peak/context. These eight readers do not establish fit for broader multi-branch
joint recurrence or 32K/128K inputs. The one held-out reader has no shuffle
contrast; eight-step losses are resource evidence, not model-quality evidence.
Use a broader held-out set before interpreting improvements. Fresh run identities
required after this code change; preserve prior attempts. User explicitly prefers
checkpointing if needed and endorses per-chain optimizer steps as a curriculum.

### Wider stress cohort and periodic recurrence probes

Owner requests filtering for larger and more recurrent samples, explicitly a wider
smoke. `scripts/select_neuralese_recurrence_smoke.py` ranks by exact crisp template
length, unfolded write sites, depth and branching, selecting round-robin across
these criteria. It preserves original splits and every source producer; reorders
selected readers first so trainer --train/--eval pick exactly that cohort. Cohort
`stress-cohort-v1`: 64 train / 24 held-out, 229 indexed producers retained,
source SHA cada05eb9c1c3d2de81501fb12ba8812499f456c4c66e9bc56ed095d1e68cbd2,
output SHA 6d891a90038fd367e87a0d5f62fe3a897e901ed6048735eb6294d7d1520f2ec0.
Extremes: 20,309 crisp reader/producer tokens, 13 write sites, depth 3, branching 7.
All 1,078 source rows have approved training and admitted trace receipts; source
splits are 1,019 train and 59 test. This is stress selection, not unbiased accuracy.

`wide-chain-v1` starts with 64 optimizer steps, batch one, depth three, chunk2048,
layer checkpointing, zero CPU activation offload, Muon, 64K admission cap (actual
cohort lengths above), checkpoint every16, held-out probes every32. Broad joint
stress remains next. Periodic eval writes eval.jsonl and restores training RNG and
stop baseline so evaluation does not change sampling. RNG restoration regression
passes (three optimizer/checkpoint/state checks). Neither run is a main quality
training run; retain receipts and report failed samples explicitly.

### Wider smoke exposes additional memory constraints

wide-chain-v1 (64 readers, depth3, no CPU offload) completed initial 24 held-out
probes and one optimizer step, then OOM in backward. Pure written/shuffled reader
CE at initialization: 1.63450 vs 1.64167 (16/24 better); this selected stress set
is not an accuracy estimate. The `soft-init.cross_entropy` field includes KL in
the current trainer; fix that labeling/objective separation before comparing it
with crisp CE. Written/shuffled fields use pure CE.

wide-chain-v2 reused v1's initial checkpoint in a new directory, changing only
activation-offload budget to2GiB (excluded from semantic resume identity). It
passed >20 steps, peak5.82GiB, but host swapping rose to26GiB. SIGTERM saved safely
and freed host memory. Do not promote that resource strategy as efficient.

Fixed checkpoint inputs: passing PortCache as an opaque Python argument kept its
tensors outside saved-tensor release/offload; cache fields are now direct tensor
inputs and reconstructed per layer. Regression ensures the cache container dies
while loss remains live, and input/weight gradients match. Trainer explicitly
deletes completed loss and collects cycles before constructing the next chain.
wide-chain-v3 (same64 readers, eval0 to isolate memory, no offload) still OOM at
step2 on teacher-program:3ed911b6030ee1ae51cf:decision:0037. Therefore graph cleanup
alone does NOT solve large depth3 chains. wide-chain-v4 with1GiB offload passed
>10 steps, peak6.97GiB, then exited137 under the9GiB host/12GiB host+swap container
limit. Large host retention remains to investigate. The bounds protected the
machine; no claim that offload is production ready.

User accepts graph splitting and accumulation. Serial accumulation already does
one backward per chain; batch1 is current minimum. Next promising alternative is
staged vector-Jacobian backward at function returns: primal payloads first, caller
adjoints, then producer recomputation/backward in reverse dependency order, with
one final optimizer update. This must preserve all input/head gradients and local
stop-boundary losses; validate against an unsplit DAG. Shorter write-depth is an
explicit curriculum alternative, not equivalent to full recurrence gradients.

## Local recurrence: exact staging and adaptive routing — 2026-10-05

Implemented `train/staging.py`: hold detached function returns, accumulate caller
adjoints, replay producers in reverse dependency order, then perform one Muon
update for the complete chain. Shared child adjoints sum before replay. This
preserves first-order gradients (branching/auxiliary-loss parity tests pass),
without retaining every producer graph or using CPU activation offload. It is
not higher-order differentiation. Stochastic stop-PG and digest-written paths
are explicitly unsupported in staged/auto mode for now.

`--backward-policy auto` estimates selected graph geometry before executing it;
executed writer/reader measurements and joint peaks refine size-bin estimates.
Calibration is checkpointed. Exceptional joint misses restore case RNG/control
state and safely retry staged, without partial parameter-gradient mutation.
`memory-routing.jsonl` records decisions and observed results. Per-case peaks
avoid a previous large graph inflating all later estimates.

Owner requests aggressive estimates: default graph reserve **0.35 GiB**, initial
geometry margin **5%**; no speculative calibration forwards. Larger envelopes
can route the same graph jointly. This is an execution choice, not a context or
language limit. A single oversized producer may still require further work.

Completed local diagnostics (original 350M backbone + older v4 port checkpoint,
not the best SFT checkpoint):
- `staged-fit-v1`: eight joint recurrence readers, 109 seconds, 4.8 GiB peak,
  no CPU offload, replay return max error zero.
- `auto-wide-v1`: all 64 stress readers, 460 seconds, 7.16 GiB peak, no CPU
  offload, zero terminal errors; two related graphs needed OOM-to-staged retry.
  Largest actual writer context 20,325 tokens. Held-out count zero: these are
  memory/gradient checks, not quality evidence.
- `auto-wide-v2`: running updated joint-peak calibration/result telemetry and
  tighter reserve, container `natlang-recurrence-auto-wide-v2`, output under
  `runs/neuralese-local-recurrence-20261005/auto-wide-v2`. Inspect before restart.

Remaining: finish v2, add meaningful held-out recurrence evaluation (correct the
mixed CE/KL reporting before comparisons), verify full converted-corpus hashes,
move from bounded stress cohorts to a representative resumable curriculum, and
validate larger single-producer contexts. CPU activation offload is not the local
default: earlier trials retained excessive host/swap memory. Keep their failures
and review holds; never promote them as successful training runs.

## Canonical machine ownership and corpus consolidation — 2026-10-05

User assigns Pop execution/resource management to the Pop agent and DGX execution
/resource management to the DGX agent. Both develop in `/home/werg/natlang`.
Procedures: `plans/MACHINE_COORDINATION.md`; root `AGENTS.md` makes inbox reads
and Git/snapshot synchronization part of the work cycle. Each machine has its
own gitignored `.coordination/inbox.md`, append/ack helper and verification
receipts. Initial coordination notes are present in both inboxes. Do not copy
inboxes over each other. User will supply DGX→Pop SSH access.

Former DGX mirror directories were moved under canonical `runs/dgx-legacy-imports`
with old-path aliases preserved. All 14 old worktree HEADs were already integrated
on main; old dirty skill-export changes are retained and their principal features
already exist in current main. These directories are historical run/runtime
provenance, not an exclusion category or another active checkout. Teacher evidence
is registered and integrated into the coverage audit/snapshot process; new work
must use the canonical repo. Active DGX Maple N2a job was left untouched.

Stopped/disabled Pop's old broad mutable development-data-sync service. New
`training/neuralese_corpora.json` gives owner, canonical path, admission, derivation
and replacement relations. Immutable per-file SHA-256 manifests are in
`training/corpus-manifests/`; `sync_training_corpora.py` handles verified transfers
without deletes or silent destination replacements. `materialize_training_snapshot.py`
freezes selected files independently of mutable run directories. S2 v2 preserves
case definitions and neuralese/tensor payloads as well as JSON traces; build caches
remain in pinned run evidence, not duplicated as training examples.

Current Pop replicas verified: S1 full final (72,036,548,160 bytes, 1,870,591 rows),
S3 derived subset (~21.8GB), v8 converted teacher (~8.15GB; fixed missing summary),
Maple v13-r2 SFT (~3.18GB), S2 selected evidence (~4.90GB/3990 files), S2 sample,
protected BGKit set, broad Maple mix, native v13-r2 corpus, selected student adapter
step13164, and historical campaign evidence (~10.34GB/29611 files). Existing data
is retained, rather than counted twice as independent examples. Native v13-r2 and
student adapter are also SHA-verified on DGX. Manifests are Git state; large bytes
and checkpoints are synchronized artifact state, never embedded in Git commits.

`training/neuralese_dataset_coverage.json` records source-ledger coverage and
remaining admission/conversion obligations. Current S1 schema audit passed, but
source-policy/provenance-alias/protected split review remains. v8 has only153
child-result writers/342 reads; rich recurrence coverage is still insufficient.
S2 candidates require paired improvement/transfer admission and modern conversion.
The source ledger still has22queued/58source-only entries; no comprehensive ready
row total is claimed.

Historical teacher audit (`training/audits/teacher-coverage-20261005.json`):
300 completed result files vs22377native teacher identities;8813accepted historical
IDs are absent from the current native ID index. This is an explicit import/alias/
version/split-review queue, not proof that all8813are distinct admissible missing
examples. DGX owner should resolve exact campaign import receipts, register a
refreshed native snapshot and convert all newly eligible trajectories. Successful
ID presence likewise does not prove digest/revision equivalence.

Local GPU: auto-wide-v2 completed64steps/473seconds, zero terminal errors, one
exceptional joint→staged retry, maxcasepeak7.071GiB, no CPU offload, exact replay.
`quality-v1` now runs128steps on64stress readers with24held-out readers and periodic
evaluation; optimizer state/checkpoints saved every8steps. Fixed evaluate CE to
exclude distillation/stop/policy penalties. At128steps held-out CE1.630→1.188;
shuffled1.191, written better on62.5%. Promising CE improvement, but only a tiny
content-use margin, not proof of strong call-return information use. Full final
summary/evaluation is still completing. This diagnostic uses original350M+v4heads,
not the best ordinary-SFT backbone. Next meaningful curriculum should use the best
student, admitted broader recurrence/skill cases, and written/shuffled controls.

Quality-v1 finished successfully (128steps,1063trainingseconds;24held-out readers),
with final written CE1.18756 vs shuffled1.19142. Full optimizer/RNG checkpoint,
heads, soft parameters, source inputs and stress-selection manifests are registered
as Pop-owned snapshots; copies to DGX are underway under canonical corpus paths.
`writer-control-v1` is now running locally: same frozen cohort/base/v4heads,512steps,
soft-context learning rate zero, normal head/control learning, periodic held-out
written/shuffled probes every128steps. This isolates gains from soft-context updates;
shared control parameters still learn, so it is not a pure writer-only attribution.
Keep it a diagnostic; it does not grant new corpus admission or replace best SFT.

Git integration: infrastructure/manifest/coverage commit8c438a71 merged the DGX
agent's new Maple plan9ac0df8c and pushed main21612b76. The follow-up adds completed
local-state snapshot manifests and exact DGX consolidation/dirty-patch review.

## 2026-10-05 recurrence corpus and inline promotion

User requested richer recurrence coverage, external providers while GPUs are busy,
removal of restrictive semantic-delegation wording, and five ad hoc layers instead
of three. Source/runtime/help/spec/tests updated; named `.nl` calls reset roots.
Details and unfinished work: `plans/neuralese/RECURRENCE_EXPANSION.md`. Current
external 32-case Space Bunny pilot uses frozen v38, keeps free-only/no-distillation
routing and retry delays. Its poor early acceptance is under review; raw failures
must remain held. Local writer-control training continues. Corpus synchronization
of the three newly published DGX snapshots was started independently.

2026-10-05 semantic folder gap: added explicit8–24file inline triage/index
families,8replay-verified pilot cases (4base/4hinted), immutable source pool
`semantic-folder-inline-pilot-20261005-v1` synchronized to DGX. Reducer prompt now
promotes semantic per-file lambdas. Existing25%reducer mix gate did not ensure
semantic child-call coverage; directory source edits and short crisp labels
must be reported separately. See recurrence expansion plan before scaling.

## 2026-10-05 reviewed recurrence expansion and policy-stage transition

Expanded bounded Clef families: skill-catalog selection and nested evidence extraction.
Reviewed packet contains 78 train programs / 52 held programs, built from 12 train
versus 8 held authored fixtures (variants are not independent semantic worlds).
`clef-rich-reviewed-cohort-20261005-v1` admits 1120 train / 747 held native
records, 670 writers, 677 producer edges, depth 5 and branching 3. Exact oracle,
source provenance, stable fixture split and producer closure all gate admission.
Six immutable registry snapshots preserve sources, teacher receipts (including
failed partial collection), admitted records, eval ordering, old-data reconversion
and task-evaluation evidence. No provider receipts themselves become SFT targets.

Conversion/5 matches canonical and runtime-rendered values using observed parentage.
Old static-lambda expansion now recovers 319 writers / 319 edges rather than the
old 28 writers; it remains a candidate subject to its existing quality policies.
Ambiguous equal outputs under the same parent remain exact text; never fabricate
a producer link. Inline instruction literals remain a later curriculum step.

Task evaluation at joint-stage step256: all five conditional return arms scored
0/4; autonomous crisp-runtime runs scored 0/4 skill tasks and 0/4 extraction tasks.
Wrong schemas/values, failure to inspect scope and unnecessary skill selection
are policy failures, not a proven channel-only problem. This checkpoint descends
from the original 350M port backbone with frozen inherited last-four-layer LoRA;
it is not the best broad ordinary-SFT student. Do not advertise capability gains
from teacher-prefix CE alone.

Course change: gracefully stop joint-learning-v2 after its step512 probe (own
written CE .958745 versus shuffled 1.096745; 24 readers, 70.83% own better).
Retain full optimizer/RNG checkpoint and best state; the original 2048-step stage
is superseded, not falsely marked complete. Next mixed stage uses the new
reviewed corpus, five-layer recurrence, trainable policy adapters and sequential
ordinary-text SFT replay. A new stage gets a fresh optimizer explicitly; unchanged
stages still resume all optimizer/RNG state. `--soft-init` carries matching soft
parameters forward; full deployment export validates pinned parent metadata.
Crisp replay restores structured write sources as typed objects and accumulates
gradients only after recurrent tapes are freed. Periodic crisp CE accompanies
written/shuffled controls; actual free-output task checks remain necessary.

Validation: 36 scoped TS tests passed; 9 Python state/render tests passed in the
pinned Torch/Muon container. New mixed GPU stage must be checked for memory fit
and convergence; zero prior errors does not establish its behavior.

Mixed-policy-v1 launched from immutable setup
`runs/neuralese-local-recurrence-20261005/mixed-policy-v1-setup/launch.sh`,
code a69a03f9. Initial held probe: crisp CE .231786, written 1.224531,
shuffled 1.247561 over 28 selected root readers. First actual update completed:
ordinary SFT and recurrent gradients accumulated; peak 1.85GiB, no errors.
Step10 peak 2.46GiB; this establishes sampled fit, not all-graph fit.
Full 2048-step stage, batch1, rank16 policy LR2e-5, soft LR1e-4, heads3e-5,
crisp weight1, distillation .25, five-layer writes, 65536 admission ceiling,
adaptive exact staging/checkpointed layers and 2048-token FFN chunks.
All1120train records retained,28root readers selected from747held (all held
producer records retained). Only one exactly matching soft piece carries over;
new task prompt/function pieces are initialized from their own text.
Current authored families have contexts around4.7k; prior stage exercised20.3k.
This richer-depth stage is not a replacement for long-context coverage testing.
Old joint stage stopped cleanly (exit0) after step540 log, preserving best512
and final optimizer/RNG state. Six data snapshots are DGX SHA-verified;
coordination note sent. Registry addition preserves the full old-stage state
and deployment warmstart as another immutable snapshot.

Follow-up: inspect mixed periodic crisp/written/shuffled probes and memory routes;
run conditional-return and autonomous held-out tasks on its saved best/final states.
Keep failures held and separate tiny authored-fixture performance from general
interpreter qualification. Evaluator now recreates expanded policy adapters before
loading a mixed checkpoint; malformed generated return arguments count as failed
decisions rather than aborting a sweep (2additional tests passed).

Exact old-stage resume on another machine requires its original path layout, not
just the checkpoint: copy registered stress-cohort records back to
`runs/neuralese-local-recurrence-20261005/stress-cohort-v1/records.jsonl`, and
registered local-recurrence-inputs pieces back to the original `inputs/pieces.jsonl`.
Parent writer-control heads are already registered at the required path.
Reconstruct the frozen package beneath `joint-learning-v2-setup/natlang_neuralese`
from Git commit e7a1e2828e11ad5bf90db61fde26812d6ee64687 (all frozen .py files
were hash-compared against that revision and match). Use checkpoint identity's
original options and pinned Docker image. Strict validation compares code hashes
and absolute resolved paths; symlinking differently located inputs does not
preserve that identity. New-stage deployment warmstarts do not preserve optimizer
state and must not be described as exact resumes. Locally, original paths and
frozen launch command remain intact; `docker start natlang-recurrence-joint-learning-v2`
would resume the superseded run, so do not start it alongside mixed-policy-v1.

## 2026-10-05 token-boundary decode fix and mixed-stage task probe

Step128 mixed probe: crisp CE .017047 vs initial .231786; written1.009458 vs
initial1.224531; shuffled1.024019;28readers. Own-better fell to53.6% so CE
improvement alone does not prove useful recurrent content. Training was
gracefully checkpointed for GPU task execution and resumed with its optimizer.

Found a real inference/evaluation bug: template decode forces text ending
`value=`, tokenized as a standalone `=`, while the trained tool call starts
with merged `=[`. Verified with the actual LFM tokenizer. Engine now forces
only the common token prefix across JSON starters and leaves the boundary to
normal decoding. The exact same step128 checkpoint then passes4/4 crisp
conditional returns; written/shuffled/zero/removed remain0/4. Old0/4 forced
conditional probes remain diagnostic evidence; do not count them as unbiased
capability scores. Autonomous execution (unaffected by this envelope) remains
0/4 skill tasks,1/4 extraction tasks. These tiny samples are not general rates.
Remaining bottleneck is recurrent value transfer and autonomous scope/skill
execution; continue full mixed training and re-evaluate rather than replacing
values with gold or declaring a solved channel.

Immutable full step128 checkpoint is hardlinked as
`task-eval-inputs/best-mixed128.pt`; active best/checkpoint files can advance
without changing this inode. New evaluation snapshot retains both biased and
corrected controls, autonomous results, deployment exports and full optimizer/RNG
state. It is evaluation evidence, never automatically SFT data.
Python regression tests cover merged boundary and malformed tool arguments.
Same decode-prefix algorithm patched in llama fork16de45539, merged/pushed
branch neuralese atb992022ab. Reference GPU behavior verified; native C++ build
and server conformance still need verification (known-working fork pin retained
with explicit pending-fix metadata). DGX agent notified to integrate there.

Close-of-pass check: mixed-policy-v1 passed256updates with zeroerrors.
Periodic256: crisp CE .023214; written .992150 vs shuffled .996472,
28readers and53.6% own-better. The shrinking control margin warrants continued
review; it does not establish useful content transfer. Mixed128 evaluation/full
state snapshot is now DGX SHA-verified (22files,1249207978bytes).
Current main and fork changes are pushed; working trees are clean.

## 2026-10-05 — foundation correction: pause compression until channel parity

User challenged the missing token-equivalent starting point. This supersedes the
recommendation above to keep mixed-policy-v1 training: the container exited
cleanly on SIGTERM at step639, preserving full optimizer/RNG state. Do not
restart compressed recurrence until the foundation gates below pass.

Parent port training did include phase B150 (A150/B150/C200/D900/E200/F300).
However B regressed shallow causal state h_k[i] onto the normalized next-token
embedding, not raw E(token_i+1), and no passed identity/channel gate was required.
Copying the full LM readout onto a shallow state does not give equivalent logits.
Existing known-text encoding also normalizes supplied embeddings and adds a write
marker; the reader normalizes again and adds two marker positions. These operations
break the proposed exact token-embedding baseline.

Controls on immutable mixed128 checkpoint:
- Valid v2: crisp4/4, raw embeddings through legacy reader0/4, known-text encode0/4.
- Valid v3: crisp4/4, raw transparent reader4/4, legacy encode with transparent reader0/4.
- v1 controls are INVALID: source selection accidentally used write name, corrected
  in v2. Keep v1 for audit, never count as capability evidence.
- Transparent task prompts still retokenize text around block boundaries, so their
  positions differ from crisp prompts; 4/4 is functional agreement, not bit identity.
- New `eval.foundation` uses spans of the original token IDs, fresh zero-residual
  content projection, raw supplied embeddings and no added marker positions. All
  28 held sources / 1875 positions have exactly equal input tensors and output
  logits. Full LM head -> selected next token -> raw input embedding is also exact.
- Loaded mixed128 shallow feedback matches full-depth greedy token choice on only
  8.32% of these positions (weighted KL5.587). This is a specific channel diagnostic,
  not a broad accuracy measurement or a proof about every older checkpoint.

Important alignment: h[i] predicts the NEXT token. Distilling h_final[i] back to
E(token_i) is a separate same-position inverse problem. An exploratory standalone
inverse fit finished4096 steps, held relative MSE .670 and cosine .556; it FAILED
and is not integrated, qualified, or a fix. `train.output_embedding_projection`
is explicitly marked experimental. A softmax embedding mixture likewise is not
identical to the embedding of a selected token, even with a copied output head.

Next prerequisites:
1. Introduce a versioned token-preserving bootstrap channel with raw input/read
   embeddings and zero content residual; keep learned legacy checkpoints explicit.
   Validate transport, positions, logits, actual typed calls and gradient replay.
2. Bootstrap causal feedback from the frozen full-depth teacher, using raw next-token
   embedding targets and teacher distribution/selected-token reference. Require held
   readout and continuation agreement before reducing depth or introducing mixtures.
3. Train compression/recurrence only after those gates. A learned projection is an
   approximation; do not claim exact equivalence merely because distillation ran.
4. Propagate qualified semantics to reference training/gradient replay/server and
   llama fork together. Current transparent reader is an internal diagnostic only;
   production defaults and old dialect semantics have not changed.

Six scoped foundation/decode regression tests pass. Visibility audit of28 held
readers finds zero visible exact source quotes and zero canonically identical
shuffle multisets;3 skill descriptions remain legitimate visible catalog inputs.
This audit alone does not establish absence of every possible information leak.
Diagnostic evidence and paused full state are registered as
`local-neuralese-foundation-20261005-v1`; admission is diagnostic only, not SFT.
DGX agent was informed of the foundation correction; DGX GPU ownership unchanged.

Storage: removed only unregistered superseded derived records/pieces, with hashes
in `runs/neuralese-storage-cleanup-20261005-v1.json`. Original native/source/provider
receipts and registered replacement snapshots remain available.

Parent control follow-up: original `inputs/port-checkpoint.pt` also fails causal feedback qualification on the same28held sources/1875positions: full-teacher greedy agreement10.13%, KL5.417. Exact raw transport/reference still passes. The weakness predates the mixed-stage backbone updates; it is not solely stale feedback from those updates. Evidence registered separately as `local-neuralese-foundation-parent-20261005-v1`.

## 2026-10-05 shared declared foundation and raw runtime handoff

Course correction: require exact full-depth causal output-state → selected
next-token input embedding initialization first. Shallow projection distillation
is an optional separately gated efficiency variant; source and context strata
must both qualify. The cutoff15 experiment's aggregate >92% does not hide its
~88% source-value agreement. Old mixed recurrence remains paused at step639.

Shared implementation: `train.recipe`, `recipes/foundation-v1.json`,
`train.port_handoff`, `eval.raw_port_handoff`, and
`plans/neuralese/TRAINING_RECIPE.md`. Default recipe now includes identity,
full-depth foundation, and actual raw runtime qualification. The DGX agent's
Maple loader/projection generalization has been merged; both machines use the
same typed handlers. The Maple cutoff12 recipe remains an optional shallow
experiment, not an inheritance of the full-depth certificate.

Full reference evidence: `runs/neuralese-full-foundation-20261005-v2` passed
132,883 held positions with exact agreement and zero KL in source/context strata,
zero optimizer updates, full optimizer/RNG state retained. The GPU runtime handoff
at `runs/neuralese-raw-handoff-20261005-v1` passed four source encode/transport
controls, ordinary greedy fixed-length writer equivalence, and finite nonzero
input gradients through actual serving replay. Its fresh stop policy and semantic
compression are explicitly **not qualified**. A raw checkpoint now records these
scopes. Raw recurrence rejects unqualified initialization; a complete declared
recurrence stage handler and broader task/stop training remain next work.

Breaking channel distinction: raw-token-v1 has no RMS rescaling or marker
positions. Legacy soft parameters are not transferred silently; re-encode source
text. Native raw GGUF export is rejected until the native runtime supports it.
Frozen projection tables are reconstructed from pinned input/output/control rows,
not duplicated in every checkpoint. The LFM norm must preserve cast-before-gain.
Single-request serving prefill/open and serving readback now project only last
position logits, reducing large-context vocabulary allocations.

Failed projection experiments are retained via the immutable
`local-bootstrap-failed-20261005-v1` manifest on DGX, with SHA-verified selected
local weight eviction recorded under `training/corpus-availability`. Explicit
`sync_training_corpora.py restore --machine pop --id ID --file RELPATH` pulls selected
registered artifacts even to their owner machine; subset receipts do not imply
full corpus availability. Diagnostic evidence is not admitted as SFT.

The post-merge scoped raw/legacy foundation, serving, replay, port and recipe suite
passes47 tests. Qualified evidence is registered as
`local-qualified-foundation-20261005-v1` for synchronization to DGX.

Qualified snapshot synchronization completed: DGX verified95files/609,757,777bytes
against manifest `cbf1b5db9b537dddf95dd23f48c5b721c132be1c7a8b0abb754c2f6927037c58`.
Additional raw/recipe tests15/15 and selected restore tests2/2 pass. Certificate
validation now resolves adjacent stage reports after snapshot relocation, retaining
exact SHA binding (the earlier original-path restoration requirement is superseded).
The entire shared default three-stage recipe is being exercised locally as
`natlang-shared-raw-foundation-v3`, alongside the cached shallow projection job.

End-to-end shared recipe v3 finished successfully (identity → full reference →
eight actual source runtime controls/replay/writer). During recurrence integration,
found one remaining raw training-only bug: the trajectory writer directly appended
and consumed a legacy open marker instead of using raw host-boundary semantics.
Fixed through shared `prefill_write_context`; raw positions now match ordinary
causal inputs. Full-depth reference feedback/native normalization stay frozen
while stop/content modules learn, avoiding needless drift from exact initialization.

Raw recurrence main stage is running (2048-step plan, Muon, full-depth raw
foundation, rank16 policy adapter, text replay, moderate2tokens/vector writes,
joint chains through depth5, adaptive staging and layer checkpointing). Startup
rejected staging+CPU-offload as expected; reran with zero CPU offload. It is using
a read-only frozen package bind mounted at the canonical package path, preserving
absolute code identities for full optimizer/RNG resume. Clean signal/resume was
exercised at step20; continued steps40/50 show no errors and writer-bearing step20
had a finite nonzero writer gradient. Frozen implementation corresponds32781162.
Do not reuse old normalized soft params; all27 pieces were initialized afresh.

Initial held losses: crisp.0601, raw soft2.1311; initial compressed written4.833
versus shuffled4.639. This is a remaining prompt/task surface gap, not a passed
semantic result. Tokenizer-only diagnostic at
`runs/neuralese-raw-prompt-diagnostic-20261005-v1` finds1–2 BPE boundary changes
in sampled held prompts (`.\n\n` text decodes identically but token IDs differ).
Measure their contribution before attributing the full score gap to boundaries.
No learned-payload substitution with literal text is used to hide this difference.
