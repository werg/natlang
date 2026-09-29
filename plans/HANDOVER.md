# Handover: training data, preference pairs, directory reducers (2026-09-27 evening)

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
