# Generation decisions — 2026-09-27 continuation

## 2026-09-30 20:04 UTC quality hold and coverage correction

- Bonsai124/314saved,72rawaccepted52reject; Luna106batchcomplete99currentadmitted/480decisions. Replenishment256workflowfreshsourcesbeingprepared; failed-source review and semantic aliases remain separate.
- Corrected priorfreshclaim:106launched =94fresh+12previouslycoveredSciFact. HistoricalscansmustjoinBOTHsource_ids/external_source.source_id; preserved earlierproofandv7/v8corrections.
- Addedcentralpendinghold3hop1__642758_643936_45121, includinghistoricgold-matchingpositives, afterindependentvisible-sourceMiddelburghomonymreview. Preservegold/history; noDPOnegative. Nodebuild/nativeholdcheckpassed. Static/recipecarryforwardpending; previousinventoryreadinessstale.

## 2026-09-30 14:32 UTC substantive Luna coverage campaign

- Started two Luna workers261656/261657 on106freshsource cases:50SciFact and56workflow,53perworker. Workflow static data already exists; model runs add strategies. Root scanned6438saved Luna rows and confirmed zero selected-source attempts. Successful same-source outcomes suppress retries even if prompt/IR revisions differ; failed-only cases need diagnosis first.
- Added explicit derivative source-identity admission guard after finding the same inherited-parent-ID bug in two preparation drafts. Corrected drafts pass; originals remain immutable. Node build/native metadata audit passed; no tests.
- Preserve superseded128case proposal and incomplete initial coverage ledger. Source-aware review and root independent launch proof correct exact-IR overcounts; ledger correction and quality-v24 publication/recipe refresh remain tracked.

## 2026-09-30 14:14 UTC Luna queue replenishment

- Resumed two Luna workers on independently reviewed signed-P&L derivative and unchanged maturity-row repair. Broader missing Luna source coverage audit underway; completed finite queues do not imply overall target completion.
- Repaired prepared derivative identity in a fresh version: synthetic source_ids and external_source.source_id must agree, parent UUID remains derived_from_source_id. Preserve earlier drafts.
- Corrected hourly reader: absent unfinished outputs are explicitly pending; absent finished outputs remain errors. Generalized canonical spool supplement to current authority queue/runtime and report baseline.24completed Bonsai results16admitted8rejected, no missing results; new rejects being reviewed.

## 2026-09-30 13:17 UTC generation status reconciliation

- Verified live Bonsai PID223773, four requests,98%GPU and fresh journal progress. Previous exhausted queue remains completed; no duplicate restart.
- Reconciled check.json to audited recipe-v23:2679selected trajectories/2129programs and1511unlabeled failure candidates. Inventory readiness is separate from final training readiness and from pending signed-P&L source review.
- Keep the prepared one-case CommaQA queue pending for the next campaign boundary rather than adding another supervisor to the active four-request campaign. No queue mutation or breaking data change.

## 2026-09-28 static corpus recovery audit

- Reconnected the two saved reference sets to staged training through frozen-runtime
  admission/materialization and an explicit joint track. 2,192 admitted trajectories,
  10,228 evidence turns, 5,174 currently approved decisions before final data gates.
- Breaking inclusion decision: six default code snapshots (120 turns) carry retired
  IR/prompts/tools and are now held. Explicit existing verified-turns inputs cannot
  bypass the IR /2 check. Preserve raw source and historical evidence; replay eligible
  saved self-contained captures instead. Descending/reverse recover 18 current turns;
  range's 20 attempts remain held for parameter writes / forbidden while.
- Corrected helper discovery to validate current file-backed projects. Fixed replay
  wrapper arguments fidelity; versioned projection function-scope-v3. Runtime rejection
  outcomes now enter replay ledgers, preventing false zero-rejection summaries.
- Inventoried 20 saved sources. Older decision/state/scene adapters remain missing;
  recorded source counts, holdouts, overlapping snapshots, sales top-up duplication
  and recovery order in plans/STATIC_DATA_RECOVERY.md. No automatic promotion of
  historical gold, synthetic reasoning, benchmark data, or archived old tool calls.
- No training or new model requests. Bonsai and single Luna repair worker remain on
  runtime-v10. Audit evidence and updated recipe: runs/legacy-data-audit-20260928/.
- Verification passed: Node build, browser bundle/types, 38 focused Node tests and
  26 Python recipe/preparation/runner tests, reference admission/materialization,
  real saved-capture replays and final 20-source inventory / recipe generation.

## Requested scope

Resume Bonsai and five Luna workers; investigate failures independently, improve agent usability, preserve data quality.
The explicit five-worker request supersedes HANDOVER's earlier two-worker limit. Allocation: folder 2, handoffs 2,
Bonsai rejection redo 1. Bonsai retains six slots. Training remains gated by the earlier explicit training instruction.

## Decisions and evidence

- Refreshed the isolated runner before starting collectors. Builds there and in the main checkout passed (Node and
  browser bundles). Runner refresh changes runtime provenance; finished compatible runs must be passed explicitly
  through `--reuse`/`--reuse-surfaces`, preserving original provenance. Do not silently relabel old trajectories.
- Added `snapshot_teacher_jobs.mjs` and used it in the redo loop: exports cover only the selected collector range,
  whereas a jobs directory can contain earlier ranges too. Three additional rejected programs were recovered from
  the 128 completed Bonsai results (redo queue 10 -> 13). Initial handover claim that exports update only at shutdown
  was incorrect: current collector merges after each completed job. Snapshotting addresses range omissions.
- Fixed admission's child attribution: a child capturing a whole collection was matched against every item's evidence
  in that collection. Identify children by instructions and argument observations, excluding captured declarations.
  Evidence explicitly in a child's instructions/arguments also counts as observed. On the fixed 31-result folder
  snapshot, 75 false missing-child-observation reasons disappear; admission stays 20/31 because real outcome
  failures remain. Regression checks include absent and unread children; both must still reject.
- No runtime tool surface or default changed yet. Avoid changing file-tool defaults before the planned measured probe.
- Found two Hotpot tasks where one retrieved article directly supports the answer but the hidden admission rubric
  requires both support articles. Keep them excluded pending a general evidence policy decision; do not erase
  evidence requirements just to increase acceptance.
- CoEdIT neutralization includes terse fragments whose need for edits is questionable (e.g. neutral headings).
  Token distance from a reference alone does not prove that an edit is needed. Keep failures excluded; inspect the
  source-selection policy before adding more neutralization data.

## Validation

Main build passed. 11 focused tests passed: curriculum admission/reference replay and atomic completed-job snapshot.
Re-audit: `runs/folder/gen1.admission.log`. No model training started.

- Added startup queue counts, provenance-mismatch notices and per-job completion logs. These are collector-only
  diagnostics, not model-visible prompt changes. Expanded collector/admission regression suite: 26/26 passed.
- Bonsai resumed with 59/75 selected results reused; the existing strict reuse checks require recollecting three
  additional older results. Luna redo reused all six existing results. Running collectors use the isolated runner;
  admission and diagnostic changes are in main and apply to audits now, collection on its next runner refresh.

## Continuation pipeline repairs

- Continuation now recognizes current `train-joint`, legacy `train-teacher`, and subsequent `train-correction`
  recipes. Rendering, audit, output paths, checkpoint initialization and original split registry follow the selected
  phase. Tests construct their recipe using the actual repository registry instead of a nonexistent temporary one.
- Materialized decisions now carry a compact final `outcome` verdict (`accepted`, `status`, optional oracle).
  The continuation validator required it but the materializer omitted it. Files and action ledgers are not duplicated.
- Materialization preserves the program's source groups as well as its own ID, so a handoff cannot acquire a fresh
  train/holdout assignment merely by getting a new task ID. Existing exported turns must be rematerialized to benefit;
  build v5 from raw trajectories, not concatenated old turn exports.
- Validation: Node + browser build; 41 focused Node tests; 10 Python improvement/continuation tests all pass,
  including a real frozen Node collector using local fixture HTTP models through prepare-correction. Tests do not
  launch GPU training. Logs: `runs/generation-{build,regression}.log`, `runs/improvement-regression.log`.

## Running and queued work

- Bonsai train4 collector PID 13007; redo follows that exact PID with 120-second intervals. Folder/handoff collectors
  use two Luna workers each; redo uses one. All use `/home/werg/natlang-runner/ts-host`, isolated from main rebuilds.
- `runs/folder/run-bonsai-probe.sh` waits for PID 13007, then runs `all`, `editor`, `files` sequentially on the same
  12 training cases (two from each of six directory families), with six Bonsai workers, seed 1201, 20 turns.
  It writes separate jobs/results/ledgers for each surface and logs to `runs/folder/bonsai-probe.log`.
  The untrained Spark comparison still needs the GPU after Bonsai work; no default surface decision yet.
- At the latest checkpoint: folder 35 completed (was 30), handoffs 68 (was 66), Bonsai 128 saved, redo 6 saved;
  collectors have live partial journals, and no `*.error.json` files in those four job directories.
- v5 assembly remains after train4/redo completes; include the new admission and materialization code in main.
  Probe and generation are asynchronous and are not claimed complete by these counts.

## 2026-09-28 morning: overnight audit and recovery

- User reduced Luna to **three total workers**, superseding five. Restart allocation: folder 1, handoffs 1, redo 1.
- Overnight: Bonsai train4 75/75 selected complete (141 saved over all historical ranges, 135 task-accepted).
  Folder 102/106 saved (70 task-accepted); updated admission is **67/102**. Handoffs 282/833 saved (116 task-accepted).
  Redo 12/13 saved; the last case exhausted rate-limit retries. Counts are task results, not all training admissions.
- Fixed a real export size limit: joining all folder trajectories into one JavaScript string failed above roughly
  512 MiB, stopped collection, and left obsolete per-job error reports even for saved results. Atomic result merge
  and snapshots now stream rows; manifest hashing streams bytes; admission reads and optionally writes one row at
  a time. Actual folder snapshot/export is 718,937,647 bytes and now succeeds; admission of all 102 rows succeeds.
- Dedicated updated runner: `/home/werg/natlang-generation-runner/ts-host`. Dependency links include both root and
  ts-host node_modules (initial missing root dependency link was detected and repaired before resuming requests).
- Handoff PID 8817 did not exit after several minutes following SIGTERM; stopped that exact PID and resumed its
  journals with one worker. Completed results were preserved. Graceful cancellation responsiveness needs follow-up.
- Bonsai probe's first surface saved 6/12, with six shared-context-limit failures. Resumed it with one worker to
  reduce concurrent KV pressure. A collector exit 2 (incomplete cases) now permits auditing and the next surface,
  instead of silently ending the entire comparison. Do not choose a default from this incomplete probe.
- Validation: Node/browser build passes; 27 collector, admission and streaming tests pass. Streaming producer
  failure preserves the prior output and removes its temporary file. Large real-data admission is checked above.

## 2026-09-28 midday: bounded Bonsai recovery; Luna stopped

- User stopped Luna entirely. Do not restart Luna or assume the earlier three-worker authorization still applies.
- Stopped probe launcher 27922 and collector 27924 (SIGTERM, then exact collector PID SIGKILL when it did not exit).
  The single-worker CUAD case had consumed hours and hundreds of child turns. Saved results/journals remain intact.
- Added collector flags `--model-concurrency` (shared across all ordinary root/child requests in a collector) and
  `--max-model-requests` (per-job new transport requests, including optional planning). Budget exhaustion leaves
  an explicit error and journal rather than an accepted training row. These flags currently do not govern a
  separately configured oracle judge; the recovery queue does not configure one.
- Added `scripts/run_bonsai_queue.py`: each case gets its own collector subprocess with a hard 600-second wall budget,
  SIGTERM then SIGKILL after 10 seconds if needed. Timed-out and failed cases are journaled and the queue advances.
  Restart skips finished attempts; reviewed retries need a new journal. Interruptions preserve partial journals.
  This is an operational collection limit; large cases are deferred, not mislabeled as model reasoning failures.
- Built `runs/bonsai-recovery/queue.jsonl` from cases with no saved result in their Luna jobs directories: 1 retry,
  1 folder case, 551 handoffs. Interleaves lanes and orders each by input size. Existing rejected results are not
  silently retried or admitted. Bonsai uses separate recovery job directories; model provenance stays accurate.
- Recovery runs on a frozen local runtime at `runs/bonsai-recovery/runtime`, independent of main rebuilds. Request
  concurrency 2, shared KV estimate 40000, request budget 128, per-call turn limit 20, transport retries 1.
  Entry exports contain only that case; gather full lane jobs with snapshot_teacher_jobs.mjs when auditing/building.
- The surface probe is deferred while useful unfinished cases progress. Its incomplete comparison cannot justify
  changing the default file tools. The large-case trace/paging behavior still needs further diagnosis.
- Validation: Node and browser build succeeded; all 15 collector tests passed outside the sandbox (its local HTTP
  fixture tests crashed under the restricted environment). Added root/child shared-budget regression, verifying no
  result is exported at exhaustion. Python watchdog regression passes: a noncooperative child is killed, the next
  case runs, and restart skips finished attempts. Live retry journal is advancing on Bonsai.

## 2026-09-28 afternoon: retired exercises and seeded handoff repair

- Replayed all seven suspicious Bonsai handoff rejections with recorded responses, without fresh model requests.
  Each reproduced the same correct answer and passed the answer oracle. The sole failing contract was the required
  seeded failure: `inline_type_repair` expects an untyped `nl` call to be refused, but that call now runs with an
  open result. These exercises were already retired for curriculum admission, but still entered handoff generation.
  Evidence: `runs/handoff-investigation/report.json`. Preserve these raw results excluded; do not relabel them as
  accepted SFT or use their correct responses as rejected preference examples.
- Centralized the retired-family policy. Admission, handoff site selection, handoff building, replay verification,
  and collection now apply it. Collection rejects a stale retired exercise before requesting model inference.
  New outcomes include named contract checks and rejection reasons so answer success and overall acceptance can
  be distinguished directly. An unmet seeded prerequisite no longer creates a wrong-result handoff.
- Fixed a separate active-exercise bug: the collector reinjected a planted failing action when the teacher was
  supposed to replace it. Handoffs now replay their opening and request the teacher replacement normally. A tightly
  scoped exemption satisfies the seeded prerequisite only when the root first-action handoff replaces the exact
  planted eval. Replaying the original rejected action must still fail before it can become a preference pair.
- Compatibility decision: seeded handoff provenance now carries `seeded_handoff_version: 2`; reuse checks reject
  older incompatible rows and partial journals. Other handoffs keep their existing reuse compatibility.
- The complete 833-entry handoff shard contains ten retired exercises; seven were in the unfinished recovery queue.
  Filtered those seven from the original 553-entry recovery queue, producing
  `runs/bonsai-recovery/queue-v2.jsonl` with 546 eligible entries before completed-attempt skips. Removed entries are
  recorded at `runs/handoff-investigation/retired-queue-entries.json`; original shards, queue and results remain
  intact. Existing timeout attempts stay deferred; they are not silently marked as model reasoning failures.
- Restarted the single Bonsai supervisor on frozen `runs/bonsai-recovery/runtime-v3`, with the filtered queue and
  existing journal. Interrupted cases resume; completed attempts are skipped. Bounds remain 600 seconds/case,
  2 concurrent model requests and 128 new requests/case. Luna remains stopped; no GPU training was started.
- Validation: Node/browser build passes; all 27 handoff, collector and curriculum regressions pass. The subsequent
  handoff-only rerun also passes after adding the pre-inference retirement assertion. The seeded regression checks
  exactly one teacher request, no reinjected failure, accepted prevention, and rejected-side replay failure.

## 2026-09-28: broad rejection review and user-directed delegation policy

- Inventoried every raw `*.result.json` under runs (76 batches, 4,434 results at the first snapshot), avoiding
  duplicated/range-limited exports. Recorded a compact evidence entry for every task or admission rejection and
  every outstanding collector error. Reusable script: `ts-host/scripts/audit-rejections.mjs`; reports in
  `runs/rejection-audit-20260928/`. New saved Bonsai results can increase subsequent snapshot counts.
- First snapshot: 907 task rejections and 1,346 admission rejections (overlapping populations), plus 585 error files
  without a saved result for their index. 553 errors were the already diagnosed runner dependency-link failure;
  the remaining categories were 17 context limits, 9 rate limits, 5 transport timeouts and 1 malformed response.
  These are collection failures, not evidence of incorrect model reasoning.
- The former runtime blanket ban on delegation inside an inline child contradicted its system prompt and caused
  thousands of failed eval actions. The user explicitly rejected that restriction. Final policy, incorporating both
  subsequent clarifications: allow **three active layers of ad hoc generated nl calls below a root**; every
  pre-existing instruction function from a `.nl` file qualifies as a fresh root. This supersedes the temporary
  proposal to carry the depth through named file calls. Existing actual-function recursion checks remain in place.
- The invocation kernel enforces the boundary; the third layer receives a distinct system prompt with ad hoc
  examples removed, an opening naming only available built-ins, and matching read_code/help guidance. Inline nl,
  Python nl and delegate share the count. At the boundary delegate is not offered. File-based shell/apply calls
  retain their source identity so their `.nl` definitions qualify as roots too. Siblings have independent counts.
- Fixed two additional confirmed runtime bugs: inherited callable namespaces were also captured, causing duplicate
  injected bindings (reproduced from handoff 65); generated functions at identical source offsets could collide in
  identity, causing false recursion. Namespace captures now use the inherited callable context once. Eval identities
  include invocation identity and source content, retaining true reentry checks for existing callable instances.
- Admission no longer rejects a successful supported answer simply because it delegated an optional field test;
  that choice becomes the `delegated_optional` note. Direct correct answers remain valid (`judged_directly` note).
  Preference building now allows an admitted direct chosen answer, subject to its rejected-side replay proof.
  Correctness, actual effects/files, honest stopping and causal evidence checks remain required. Sampling can still
  select delegation examples using facts/notes or the explicit --require-technique option.
- Histories showing the removed blanket ban or internal duplicate-binding fault are now marked obsolete for
  admission, even when the eventual answer was correct: training them would teach an unavailable runtime rule.
  Raw records are preserved; do not relabel or promote them blindly. Current-policy re-audit identifies these rows.
- Improved a common loop diagnostic with a concrete finite pagination form that computes the page bound once.
  While/recursion policy is unchanged. Other remaining failures include genuine FOLIO/ANLI judgments, invalid
  generated code, partial file edits/CSV extraction, source-oracle edge cases and large-case budgets; see review.
- Compatibility decision: `execution_policy_version: 2` is part of provenance and reuse checks. Incompatible old
  partials/results are not silently reused under the new delegation/capture/identity behavior. Saved raw results stay
  in their original jobs directories. The retired-family and seeded handoff version 2 policies still apply.
- Froze the built code as `runs/bonsai-recovery/runtime-v4`. Stopped supervisor PID 138523 by exact PID to migrate;
  interrupted handoff 589 remains unfinished and will resume/recollect under the new provenance. New queue-v3 has
  six reviewed probes in separate `system-fix.jobs` before the existing filtered recovery entries (552 total before
  journal skips). Probes cover ANLI, captured facts, FOLIO child calls and relational pagination; model is Bonsai,
  including probes sourced from old Luna failures. Luna was not restarted. First ANLI probe finished in 71 seconds
  with a real answer rejection, which remains excluded; the next case is advancing. No model training started.
- Validation: Node/browser build passes; 123 focused interpreter, compiler, runtime, collector, handoff and admission
  tests pass, plus 14 compiler/folder/migration tests. The 60-test interpreter rerun also passes with identical source
  text in successive ad hoc layers. Tests cover all three layers, fourth-layer rejection, prompt/help consistency,
  fresh file-root budgets, inherited namespace calls and correct direct/delegated admission.
  Browser type checks and the raw-job audit regression also pass (138 distinct focused tests in total). Frozen
  runtime-v4 src/dist hashes match the reviewed main build. At the live checkpoint, 12 of queue-v3's 552 eligible
  entries have finished attempts (including earlier deferred timeouts); 540 remain. Probe 65 has advancing saved
  turns, and Bonsai health is OK. The machine audit's successful-run mix is reported in REJECTION_REVIEW.md.


## 2026-09-28 — Slow-case evidence and data quality version 2

- Investigated the saved timeout replies, separating fresh inference from replayed prefixes. Handoff 65 genuinely
  churns through rejected premise combinations (13 fresh replies / 11,208 tokens). Its checker rejects every set
  in the removed-premise variant, while its child assumes support exists. Removing one annotated EntailmentBank
  premise does not prove no alternative science proof exists: quarantine these source variants, block their
  handoffs/preferences and stop generating them. Preserve raw evidence.
- FOLIO 142 and 217 each decode about 12,000 tokens with finite model checking and completed child judgments.
  They make progress with considerable rework rather than sitting in a frozen request. Folder 84 is an execution/
  context stall with only one per-file judgment. TextWorld 6 explores but its generated step returns stale
  pre-action state. Full per-case evidence is in the slow-cases artifact; do not label all timeouts bad reasoning.
- General fixes: strict CSV parsing/schema/unique ids, separate positive recall/precision, literal per-contract
  quote provenance, independent semantic grading for alternate quote extents, strict count-line parsing, actual
  report/return consistency, INDEX totals and per-original-file move scoring. Read-only corruption is a hard
  failure. Word overlap no longer certifies rewrites. Alternate/unchanged CoEdIT drafts need an independent
  rubric verdict; tasks say review/edit as needed and count actual edits.
- Source fixes: quarantine conflicting labels instead of last-row overwrite; key CoEdIT splits by visible draft
  across task/instruction wording and retain targets; retain Hotpot supporting sentences, group by visible
  question and quarantine conflicting answers; retain CUAD alternate spans and quarantine unusable positives
  rather than calling them negative. Generator reports now include source quarantine reasons/identities.
- Breaking evidence/admission decisions: reference child layout/markers and technique/efficiency differences
  become notes. Correct direct/delegated/mixed results can be admitted; genuine result/effect/file/repair contracts
  and follow-up causal checks remain. Hotpot replaces two hidden article-lead markers with observed annotated
  supporting-sentence evidence. QA article normalization/numeric checks and source-aware rubric fallback handle
  equivalents; date-shaped gold leads to explicit calendar-date wording. No unchecked substring equivalence.
- Breaking training decision: partial file items and below-perfect aggregate agreement can be useful evaluation
  results but are held for review before training their constituent judgments. Missing/insufficient grading is
  quality_pending, excluded from SFT and negative handoffs/preferences. Technique-only preference pairs are
  retired. The standard SFT builder no longer defaults to excluding direct solutions. Legacy folder contracts
  and already-prepared old SFT/pair artifacts require fresh admission/rebuilding before training.
- Admission is version 2; data_quality_version 2 and judge identity/endpoint hash participate in reuse. Teacher
  and judge must have distinct model IDs. Grading shares the whole-case request/concurrency/KV budget, uses
  bounded contract context, and supports needs_review. No independent judge was started; ambiguous free
  rewrites and alternate quoted extents remain excluded pending adjudication, not relabeled model mistakes.
- Supervisor now journals activity every 30 seconds and distinguishes inactivity_timeout after five minutes
  without a saved reply from the hard case deadline. Default remains 600 seconds / 128 requests; FOLIO/ANLI
  batches get 1,200 seconds; all-file tasks get size-based budgets up to 3,600 seconds / 512 requests. Request
  concurrency remains 2, collection workers 1 and turns per call 20. Repetition counters are diagnostic only.
- Built seven reference-verified replacement cases across six folder families (six distinct tasks; one hinted
  twin). queue-v4 contains 558 entries before skips, including two measured FOLIO retries; removes the unsupported
  premise probe and legacy CUAD case. Migrated from supervisor PID 154289 by exact PID to frozen runtime-v5.
  Handoff 591 was interrupted and remains pending in its original raw directory. Fresh v5 jobs use distinct paths.
  Supervisor PID 181397 owns the new queue; first task is quality-v2:0. Luna remains stopped; no training started.
- Both relational probes are task-accepted (171.5s / 205.9s); handoff 589 also finishes correctly in 427s.
  New snapshot: 4,443 raw results, 909 task rejections, 1,464 admission rejections and 585 collector error files.
  Admitted raw-run technique mix: 2,087 direct, 532 inline, 213 named-only (not deduplicated final training counts).
  Queue checkpoint: 16 finished attempts and 542 pending. Preserve source grouping and held-out separation.
- Validation: Node/browser build and browser type checks pass; 132 focused regressions pass, including CSV,
  rewrite/quote grading, pending verdicts, shared judge budget, time-vs-inactivity bounds, source conflicts,
  mixed delegation, admission, runtime/compiler/files/handoffs and raw-job audit. All seven replacement references
  verify. Frozen runtime-v5's 332 src/dist file hashes match the reviewed build. New runtime/prompt retains the
  user's three ad hoc layers per pre-existing .nl file root.
- First fresh replacement checkpoint: the 39-message move task completes in 369.3 seconds (11 fresh replies /
  5,732 tokens, two repeated action sets). It moves all 39 files and returns the correct count, but eight go to the wrong label folders;
  the file-placement contract rejects it. This is incorrect classification, not a stalled server or
  false oracle rejection. The queue has advanced to the 22-message INDEX task. No partial result was admitted.
- Correction to the first live diagnosis: failed move-item names are original source identities, not paths where
  files remain. Actual output has an empty inbox; eight destination labels disagree with gold. The count is 39,
  not 31. Corrected the log after inspecting actual paths; the source files and classifier rejection were preserved.
- Repair generation exposed a real replay defect: recorded() dropped prompt/completion token counts. That changed
  context calibration and the timing of forced compaction, so a valid source failure was refused because replay
  observations differed. Preserve usage in recorded turns/handoff prefixes. The complete saved SMS run now replays
  with identical observations and the same classification rejection; its repair handoff verifies. No relaxed replay
  comparison. Range exports are still selected-range views: use a raw-jobs snapshot when building repair shards.
- Final live checkpoint: INDEX had 50 saved replies / 11,705 completion tokens before its 670-second deadline.
  Exact-request offline replay shows a completed 22-label batch followed by a full repeat; many identical answers
  were legitimate sibling classifications. Folder guidance now offers direct or delegated judgment, reuse of
  completed results and focused disagreement checks. This remains guidance, not a required technique.
- Rewrite completes in 575.5 seconds: 13 files actually changed and returned count 13. Twelve outputs need semantic
  adjudication; held out rather than negative labels. Article QA completes in 48.6 seconds with Dulce River vs Dulce,
  held for source-aware equivalence. Rubric-backed answers with zero lexical overlap also require review.
- Verified one SMS repair handoff against the updated build. Migrated by exact supervisor PID 181397 to queue-v5
  and frozen runtime-v7 (source dee31c3; all 452 manifest hashes verify). Runtime-v6 is an unused intermediate
  snapshot; frozen snapshots are immutable. Interrupted extraction raw partials preserved; fresh jobs use new paths.
  New supervisor PID 213073 prioritizes quality-repair:0, then resumes pending generation. At migration: 559 entries,
  20 completed attempts, 539 pending. Luna remains stopped. No independent grader or model training started.
- Validation after replay/oracle fixes: Node/browser build and browser type checks pass; 45 focused tests pass.
  After folder-guidance change, builds/type checks and 17 folder/handoff/oracle tests pass. Reviewed all eight SMS
  mismatches against actual source text: clear personal vs prize/premium-service distinctions; no contradictory gold
  found in those items. Raw histories and exclusions remain the authoritative audit trail.

## 2026-09-28 Bonsai status checkpoint after queue-v5 migration

- Supervisor PID 213073 remains live on frozen runtime-v7; server health is OK. Queue-v5 has 23 finished attempts
  (18 collector completions, 5 deadlines), 536 pending including the current FOLIO repair 142. Completion does not
  imply task correctness or training admission. Luna remains stopped; no training started.
- SMS repair finished in 147 seconds but preserved all eight wrong destinations. It checked counts rather than
  re-evaluating contents; replay repair correctness is fixed, but this handoff did not improve the task result.
- CUAD extraction hit its 880-second deadline after 53 fresh replies / 13,759 completion tokens. The last saved
  steps show clause extraction/verification and preparation for the full batch, not frozen transport. Raw partial
  remains available; no timeout negative or partial output is admitted.
- Mixed payment task finished in 354 seconds with 4,064 against 4,708 and is excluded. Its trace shows handwritten
  amount transcription and repeated aggregate code, plus uncertain category choices. Investigate source-linked
  bookkeeping separately from category semantics; do not describe this as a server stall.
- Pending independent rewrite/answer-equivalence judgments remain held out. These latest fresh probes have not
  produced an accepted training row; infrastructure is progressing, task quality still needs work.

## 2026-09-28 single Luna repair worker authorized and started

- Latest user request explicitly restarts one Luna worker, superseding the earlier stop instruction. Started
  gpt-6-luna on openai-codex from frozen runtime-v7, PID 237444, one collection worker / model concurrency 1,
  including children. 16,384 context tokens, 20 turns per call, 256 whole-case requests, low reasoning and
  execution plans. Bonsai supervisor PID 213073 continues. No model training or independent grader started.
- Current raw audit: 4,448 results, 914 task rejections, 1,469 admission rejections, 585 collector errors.
  These are historical attempts, not unique unresolved tasks. Deduplicated by original program ID, removed
  already admitted/solved cases, retired/unverified contracts and quality-pending rows. Repeated handoffs collapse
  into root cases. Luna reruns complete tasks, so earlier wrong file placements can be reconsidered.
- 157 potential unresolved cases: 152 task-failure candidates and 5 collector/deadline-only candidates. Launch
  preflight found 12 legacy IR records; the existing migration upgraded 9 and validated their root definitions.
  Three source-case records have no convertible lambda root and are preserved in deferred.jsonl for reconstruction
  (order_saga:3, shopkeeper:0, webserver:0). The runnable queue is 154: 149 task failures and 5 infrastructure cases.
- First launch stopped before inference because the whole shard contained legacy IR. After migration/filtering,
  all 154 records load and the collector is live. Provider prepare confirms credentials and exact Luna model
  selection. Source/selection/deferred files and summary are in runs/luna-repair-20260928/; the audit is in
  runs/rejection-audit-20260928/luna-repair-current/. Do not regenerate the shard while this collector is active.

## 2026-09-28 repair-worker recovery and yield checkpoint

- Luna completed 3 tasks: 1 accepted relational multihop repair (also passes current training admission), 2 wrong
  answers (late-bound grants and payment sum). Two other cases reached the 256-request whole-case budget and remain
  errors/partials, not negatives. Then a ScienceWorld spawn ENOENT killed the collector: frozen runtime-v7 resolved
  vendor beside the frozen directory, where no dependency mapping existed, and WorldBridge lacked an error listener.
- Freeze setup now maps the repository's vendor dependency tree beside frozen runtimes and refuses a conflicting
  mapping. Shared vendor symlink repairs existing Bonsai/Luna runtime-v7 paths without changing frozen source files.
  WorldBridge now rejects and clears pending requests on process error/exit instead of an unhandled process crash.
  Main Node/browser build and browser type checks pass; existing world-service declaration regression passes.
  A missing-interpreter check now rejects the case with ENOENT while the collector process stays alive.
- Luna briefly resumed on repaired v7, then migrated by exact PID 278166 to frozen runtime-v8 (source c024b10;
  manifest hashes verified). One worker / one request in flight remains enforced. Current PID 278800 skips its
  three completed jobs and resumes the remaining 151, with saved partial replies preserved.
- Bonsai remains live on runtime-v7 / queue-v5, server health OK. Checkpoint: 42 finished attempts / 517 pending.
  Seventeen recent repair results pass task contracts (including FOLIO 142); this does not certify all as final
  training rows. Handoff 802 fails; hard deadlines and quality-review exclusions remain distinct from model negatives.

## 2026-09-28 improving new failure patterns (runtime-v9)

- Investigated Luna budget failures: payments7 has 142 messages. A planning/action pair per child needs at least
  284 sends before root work, so the old 256 cap was insufficient. FIFO request admission also queued all sibling
  plans before actions, and counted waiting requests; cap exhaustion cancelled previously admitted work. Teacher
  transport now holds one concurrency slot across its plan/action pair, charges requests after capacity gates,
  and rejects later requests without cancelling earlier admitted work. Regression saves two child actions at a
  six-request cap, then resumes to acceptance. Request counting still includes plans and independent judging.
- Serialized partial checkpoint writes: concurrent atomic renames were not ordered and could overwrite a newer
  snapshot. No claim that a particular observed failure lost its responses; the unsafe ordering is removed.
- Grants failure is unintended shared mutation: predicate calls decremented/reset captured budget. Openings now
  explain sibling/caller write effects and instruct judgment calls to use fresh calculation locals unless the
  task requests updates. Mutable captures remain functional; delegation remains optional with the same three
  layers per pre-existing file root. Live retry currently uses fresh locals and has not repeated those writes.
- Payment contract change: BANKING77 positive labels include unrecognized direct debits and ATM cash withdrawals,
  while the old prompt said only charge. Explicit scope now includes card/direct-debit/cash withdrawals, duplicate
  charges and extra charges. payment_scope_version=2 required for BANKING77 folder_mixed; old ambiguous contracts
  are held out. Fourteen Luna repair records migrate under :scope-v2 IDs and recorded generation provenance;
  unchanged source grouping/gold, no label leakage. All 14 updated reference replays verify.
- General folder guidance pairs semantic judgments with original IDs and parses the amount table instead of
  retyping it. Large files should be passed by handle and searched/read in the child, with short NL instructions.
  The prior contract probe embedded a 221 KB file in instructions; guidance now addresses that pattern.
- Reused the bounded queue supervisor for providers: one Luna collection worker / one request in flight,
  execution plans, 20 turns/call, base 256 sends with 4 sends/item + 40 root allowance (ceiling 1024; actual caps
  256–748). Case budgets 900–3600 seconds; FOLIO/ANLI 1200. Existing five-minute saved-reply inactivity guard retained.
  No deadline or incomplete result becomes a negative label. Earlier raw jobs/partials remain preserved.
- Frozen runtime-v9 (source ce70dee; all 452 manifest hashes verified). Luna migrated by exact PID 278800 to
  supervisor PID 286423, queue-v2.jsonl / v2.jobs, 153 selected (already admitted case 116 excluded), three legacy
  reconstructions deferred. First retry is grants case 106 to measure the shared-budget fix. Bonsai migrated by
  exact supervisor PID 213073 to PID 287222, queue-v6 and fresh .v9 job paths, same append-only journal; 559 entries,
  46 finished attempts / 513 pending checkpoint. No pending Bonsai payment task needed scope migration.
- Validation: Node/browser build and browser type checks pass; 100 focused tests pass (collector scheduling,
  durable progress/resume, judge budget, interpreter/captures/depth, handoffs, oracles, folder references, supervisor
  time/inactivity behavior, single-Luna provider configuration, audit). No model training or independent grader started.

## 2026-09-28 runtime-v9 live checkpoint

- Both supervisors remain live (Luna PID 286423, Bonsai PID 287222); Bonsai server health OK. Luna v2: 4 completed
  attempts / 149 pending including the active large payment case. Three task-tolerance passes, but only two pass
  current training admission. Keep task acceptance and data admission separate.
- Grants17 now returns exactly the expected independent budget judgments and passes admission in 673.3 seconds;
  the previous Luna run corrupted shared budget and failed. SMS move now places all 39 files correctly and passes
  admission in 297.2 seconds, repairing the eight earlier destination mismatches.
- SMS INDEX completes in 227.7 seconds, returns/reports ham=16 and spam=6 against gold ham=18/spam=4. Benchmark
  agreement tolerance accepts it; current admission excludes it as quality_pending_partial_agreement. Do not claim
  this is a complete repair or include it in SFT. Remaining label errors need review.
- Payment0 completes in 333.1 seconds with 3161 against 4708; excluded. It now parses source CSV and joins included
  IDs rather than retyping amounts. Remaining semantic classifications still need investigation/adjudication.
- Large payments7 is active with 72 saved model replies after 480 seconds, versus only two saved replies when the
  previous 256-request run exhausted its cap. Current cap 608 / deadline 3600 seconds. Repeated yes/no child actions
  are sibling judgments, not evidence that the parent is repeating the whole task; do not auto-classify as stuck.
- Bonsai checkpoint: 52 finished attempts / 507 pending; last five observed repairs pass task checks. These remain
  raw-result counts; not deduplicated or universally certified training rows. No new Luna collector error files
  in v2.jobs at this checkpoint, and no model training or independent grader has started.

## 2026-09-28 payment criterion and source-review holdouts

- Payment0's 4708 versus 3161 is explained exactly by three omitted positives (629 + 644 + 274 = 1547).
  The parent says "message is about" an issue; the child added "report" and "exclude general questions".
  This wrongly excludes the strange £1 transaction question. New generic delegation guidance preserves the
  parent's criterion/context and forbids stronger evidence requirements or added exclusions. It lives in the
  delegation paragraph, which is removed from the third-layer prompt. Delegation remains available/optional.
- Two cash examples do not establish an unrecognized withdrawal from the visible wording. The large payments7
  case then completed: 142 judgments, one mismatch, 13386 versus 13888, in 1060.5 seconds with 162 saved replies.
  Its remaining example asks how to dispute a debit transaction; the label assumes an unrecognized direct debit.
  These are source-review questions, not verified wrong labels or evidence that the model must learn to guess.
- Course change: a shared, explicit source-review registry holds these three source inputs pending adjudication.
  Current text-based IDs and old text+label aliases both match. Future labeled-row generation excludes pending
  sources; shared quarantine policy holds existing affected tasks out of admission, negative handoffs and pairs.
  Original dataset labels/gold and raw results stay intact. Resolution requires recorded independent evidence;
  no independent grader or model training has started. Do not change gold to a teacher's preferred answer.
- Benchmark tolerance still accepts the large total, but strict training admission excludes partial agreement.
  Current Luna checkpoint: five completed attempts, two admitted, two tolerance-only partials, one failed sum,
  148 remaining; no v2 collector error files. Bonsai: 53 attempts finished, 506 remaining, 46 collector completions
  and seven historical hard timeouts. Both supervisors keep running on frozen v9; main source-review policy is
  available to export/admission now, while prompt/collection changes await a safe frozen-runtime migration.
- Validation: Node/browser build and browser type checks pass; 38 focused source-policy, generator, oracle,
  handoff/replay and collector regressions pass. Per-ID payment mismatch artifacts are saved under
  runs/rejection-audit-20260928/payment-followup/; small-case duplicates are removed before summing.

## 2026-09-28 runtime-v10 rollout

- User authorized rollout. Froze source d69af3a as runtime-v10 and verified all 455 manifest hashes. The new
  runtime contains parent-criterion guidance, pending-source quarantine and generator filtering. The final
  ad hoc layer still removes delegation guidance; the three-layer runtime boundary remains unchanged.
- Filtered source-review tasks from the queues: Luna v3 retains 150 entries (three held); Bonsai v7 retains
  558 (one held). Held-entry audits retain source/index/IDs and reason; no source/gold rewriting. Existing
  append-only journals retain completed-attempt skips. Completed and boundary entries retain their old artifact
  paths; pending entries get .v10 paths because old prompt histories must not be reused silently under new prompts.
- Waited for exact active cases to reach their existing bounded outcomes, then stopped old supervisors by exact
  PID, waited for old child processes to exit, and started new supervisors. A successor briefly launched by an
  old supervisor is stopped during handover; its raw artifacts stay preserved. No concurrent Luna collectors.
- The boundaries were hard timeouts, not successful repairs: Luna extraction 0 saved 106 replies and had written
  a 29-contract CSV with a reported 10 clauses, but was still checking/repairing verification at 900 seconds
  (908 including shutdown). Bonsai handoff 555 saved 18 fresh replies and hit 600 seconds (603.1 with shutdown);
  it was attempting lengthy amount-parsing code, including a forbidden while loop. Both checkpoints remain
  unfinished, excluded from training and model-negative labels. Review retries/budgets rather than promote them.
- New Luna supervisor PID 332686: queue-v3, v2 journal, one worker/request including plans/children; starts case 4.
  New Bonsai supervisor PID 332210: queue-v7, existing journal, starts handoff 552; server health OK. At migration,
  eligible queues have 146 and 504 remaining, respectively. Rollout audit, provenance and PID state persist in
  runs/bonsai-recovery/runtime-v10.*. No model training or independent annotation grader has started.
- Post-rollout progress verified: Luna's new case saved 17 replies and Bonsai's new handoff saved six fresh
  replies. Both are using new .v10 checkpoints; these observations establish live requests, not task acceptance.

## 2026-09-28 source research and offline trajectory route

- User requested new task sources and existing trajectories convertible to our IR without teacher generation.
  Primary-source shortlist and sample findings are in DIRECTORY_TASK_SOURCES.md; research samples/summary
  live under runs/source-research-20260928/, outside admitted training data.
- Added an offline route to the roadmap: start with successful structured Nebius OpenHands / SWE-smith traces,
  then NVIDIA for diversity. Conversion must restore initial state, preserve action semantics and evidence,
  replay deterministically and retain separate source-success versus native-validation provenance. Candidate
  counts are not admitted volume. Unsupported mid-trace tools and unknown success cannot be hidden by rewriting.
- New directory task priorities: WorkBench state changes, CommitPackFT edits, TAT-QA table/document reasoning,
  then MuSiQue evidence chains. FinQA unit consistency and QMSum judging remain review work. Commit snapshots
  are not full trajectories; source demonstrations must not fabricate reasoning or delegation.
- No external cases imported, new source generation started, benchmark boundary changed or runtime modified.
  Workspace-Bench/MuDABench remain evaluation-only. Research caught mock AgentSynth positives and stale
  trajectory counts; explicit success and source terms need checking before admission.
- Existing runtime-v10 supervisors remain live: Bonsai PID 332210 progressed to handoff 790; single Luna
  PID 332686 finished case 10 (175.6 seconds, 20 replies) and started case 11. Collector completion is not
  training admission. This documentation-only checkpoint requires no implementation build/test run.

## 2026-09-28 source-backed static pilot and training wiring

- Built a bounded pilot from actual pinned/source-hashed data: WorkBench 8, CommitPackFT 12, TAT-QA 11,
  MuSiQue 12 and SWE-smith 8: 51 native directory cases, 157 approved decisions, 32 held direct-result
  decisions. Source acquisition and reference replay use zero model calls. The exact Sharp tokenizer/template
  audit admits all 157 at 8192 tokens, largest 6356; 8442 supervised tokens after synthetic-note masking.
  Results/IR/manifests and concrete rejection ledgers are in data/teacher/source-backed/. Final render/audit
  artifacts live under runs/source-backed-20260928/verified/. See DIRECTORY_TASK_SOURCES.md.
- Course decision: CommitPack's vague commit descriptions are not unique gold specifications. Scoped tasks
  expose an explicit change request derived from before/after content. WorkBench retains every matching
  sender record, projects irrelevant email bodies, checks latest-date gold and preserves untouched files.
  TAT-QA rejects inconsistent/unsupported derivations and reads evidence before choosing its calculation.
  MuSiQue uses original train data, excludes 2768 dev seed IDs and groups shared seeds/paragraphs.
- Trajectory decision: 120 captured rows include 48 parsable known-success runs. Complete SWE traces are
  still incompatible (shell/install/test/environment dependencies). Import eight successful file-creation
  operations as separately specified, self-contained tasks; retain unchanged payloads/source observations,
  pinned repository license evidence, original IDs and row hashes. Mark native wrapper steps and scope;
  do not claim whole-issue replay or fabricate model reasoning/delegation. Nebius needs base-commit joins;
  NVIDIA's sampled rows have unknown outcomes. Raw failed/unknown evidence remains excluded.
- Pipeline fixes: preserve source licenses, IDs, revisions and gold attribution in native turns and both
  renderers. The staged renderer now masks scripted notes; token auditing counts masked notes as context,
  matching trainer behavior. Converted-source admission/materialization checks task/outcome/trajectory
  bindings and source scope, with explicit rejection on tampering. Ordinary existing rows retain their path.
- Connected the static bundle to both entry points: automatic checked-manifest inclusion in build_lora_sft.sh,
  and a frozen validate-static-sources stage feeding the staged pipeline's coding lane. Explicit disable/select
  options are available. Generated runs/source-backed-20260928/recipe.json without launching it. The pilot
  ready artifact is bound to a tokenizer-only Sharp template profile; production re-renders for its selected
  model. No model weights loaded, no training or new teacher queue started. MuSiQue's default static SFT
  contributes retrieval rather than invented reasoning; real reasoning/answer trajectories need collection.
- Validation: Node/browser build and browser type checks pass; 41 focused Node checks, 30 Python checks,
  shell syntax and actual acquisition/replay/render/token audit pass. Live Bonsai PID 332210 and the single
  Luna PID 332686 continue frozen-v10 work; latest check showed progressing handoff 625 / Luna case 25.
- CPU-container export artifacts initially had root ownership; normalized ownership to the workspace user
  and removed superseded pilot exports after verifying the final artifact. MuSiQue acquisition now stages
  downloads atomically and retries incomplete archives. Static builders require newly built main/new training
  freeze; the old generation freeze does not contain the source-conversion helper. Do not mutate live v10.


## 2026-09-28 — modern static adapters and conservative admission

User requested modern recovery, only high quality training data, and audit of already
integrated static sources. Implemented seven source adapter families, full raw
SCONE/SGD/CLEVR joins, separate holdout/held libraries, and a 72-case native pilot
connected to both builders. A replay cannot override held source quality. FinQA units,
sales/Typed labels, semantic synthetic programs and 620 modern leaf joins stay held.
FinQA dataset license corrected to CC-BY-4.0; no FinQA case is trained.

Initial documentation-route discrepancy was my adapter's `docs`/`documentation`
mismatch, not bad source gold. Corrected it, verified rubric values, regression tested.
Added fold-step definition recovery rather than guessing leaf signatures.

Rebuilt 51 directory cases with identical IR and replayed all 2,192 static references.
Sharp token audit found 661 excess reference pairs. Breaking final-data change:
`audit_training_corpus.py` now removes rendered duplicate pairs, keeps attribution and
the most restrictive reasoning mask, and holds train/holdout-connected components.
Old ready audit artifacts must be rebuilt. Scripted/source replay evidence is reported
separately from generated teacher evidence. See MODERN_STATIC_ADAPTERS.md for commands,
counts, artifacts, test limitations and remaining review work. Live Bonsai and single
Luna workers were not migrated or restarted; no model training was launched.


Final combined Sharp audit: 5,656 rendered static candidates -> 4,995 unique ready
decisions, removing 661 duplicates; zero train/holdout-connected rows in this mix;
max 6,356/8,192 tokens. Explicit test verifies recovered original holdouts cannot be
promoted by an ordinary teacher result even if its current split is relabeled train.
Canonical bundle is data/teacher/recovered; superseded intermediate builds were removed.


## 2026-09-28 evening — rejection contracts and conservative source holds

See REJECTION_FOLLOWUP_20260928.md and runs/rejection-followup-20260928-evening/.
Audited 78 final v10 results and seven unfinished attempts. Fixed false observation
requirements for prevented planted compile failures, misleading eval rollback text,
mail declaration/changed-key retries, highlighter iterators, cross-store CommaQA
contracts, real reference arithmetic and ALFWorld turn budgeting. Full numeric
audit verified 25,098 source steps after correcting our initial signed difference
to the source's absolute-gap semantics. No gold labels changed.

Breaking decisions: hold six ambiguous αNLI inputs and legacy numeric contracts;
rebuild affected ready artifacts. Changed snapshots preserve source identity but
restart incompatible handoff prefixes, with explicit migration records. Mail
rejects changed contents under a delivered key. Priority queues add eight Bonsai
and eleven Luna reviewed retries, with one Luna worker/request. V10 stays immutable;
v11 rollout records completed case boundaries and actual new PIDs. No training.

Later results exposed FOLIO English/formalization mismatches in stories 337
and 162 (negated-XOR mistranslation; invitation turned into performance). Hold
both stories across single/batch adapters, retaining original annotations. Luna
started its highlighter retry on v11 after its boundary. Freeze v12 for the new
holds and prune queue-v9 / queue-v5; finish progressing cases before migration.
Final focused validation: 110 Node tests, two queue tests, browser build/types.

Final follow-up: native materialization now honors holds even for direct/failed-run
exports. The highlighter retry exposed stale gold roles and renderer whitespace;
hold legacy highlighter oracles until independent v2 verification. Source freezer
v3 filters four held seeds (36 eligible cases / nine codebases) with a ledger.
All 2,192 integrated static references still admit; 5,174 approved and 5,054 held
decisions before dedup. Export/gate validation passes 127 focused Node checks.
Freeze v13 and queues v10/v6 for final rollout. Stop only the already source-held,
repetitively failing Bonsai handoff 23 with a quality-hold event, not a negative.
Other valid progressing cases finish before migration. First ALFWorld retry reaches
score 100 in 22 moves. No training, source relabeling, or additional Luna workers.

## 2026-09-29 — explicitly released WorkflowEvals and visible source evidence

User released the four retired WorkflowEvals test repositories into training and
confirmed Apache-2.0 for all. Keep original test provenance; record user license
confirmation separately from upstream licensing. Breaking decision: a narrowly
pinned retired-evaluation exception permits exactly these four revisions, which
must no longer serve as held-out evaluation. No general source-test promotion.

Import model-generated typed labels only with two-provider >=0.95 modal agreement,
validated distributions/recomputed consensus and bounded original inputs. Preserve
8,911 held questions, never relabel gold. Score tasks classify the original string
ordinal index, not the weighted expected score. Selected directory batches do not
claim complete workflows; input agent traces are review evidence, not successful
demonstrations. 4,194 judgments plus 611 batches pass native replay.

Breaking answer-policy decision: verified static conversions from these exact
sources may train typed direct answers with synthetic action-note reasoning
masked, rather than inventing reasoning or silently dropping the source answers.
Other scripted sources retain their existing direct-answer policy.

Trajectory inspection found the first directory reference discarded readText
results, leaving input evidence out of the final context. Withdrew that draft;
rebuilt using visible read_file and every read_page, plus independent complete-input
reconstruction. New bundle has 4,805 cases / 8,453 approved decisions / zero
unlinked. Both training-data entrypoints discover it; default rematerialization
matches native turns. Final model-specific token/rendered dedup audit remains a
training-build requirement. See WORKFLOWEVALS_IMPORT.md and run audit artifacts.

Acquisition OOM fixed by sequential 16-row Parquet batches (1GiB). Superseded task
scratch replays are losslessly compressed with verified decompressed hashes to
recover disk space. Existing Bonsai v39/queue-v31/evaluation watchers were left
running unchanged. No model training and no new Luna workers.

Follow-up quality concern: older MuSiQue/QASPER/SciFact static references also have
some discarded-readText actions. Their unsupported scripted final answers were
already held by the existing direct-answer policy; do not enable direct answers
for those bundles. Rebuild them with visible evidence before admitting conclusions.

Final provenance improvement: joined checksum-verified scenario metadata and grouped
security counterfactuals by upstream activity, customer turns by dialog identity.
Replayed all4,805 again on frozen runtime-v2; default8453-turn materialization matches.
Current teacher holdout pools share zero source IDs/groups with this new bundle.
The prepared64-case Bonsai pilot has16 reducers (25%); it is not in the active queue.
Static mix audit before model-specific rendering:12,486 approved decisions,7,925
reducers (63.47%); this is supply coverage, not the final token-filtered training mix.


## 2026-09-29 — user requested two Luna generation workers

Supersedes the one-worker limit. Started22544/22545 on disjoint32-case queues from
WorkflowEvals teacher pilot, each24 primitive/eight directory roots; total64 cases
and25% directory reducers. One model request per process means two total. Both
use frozen runtime-v2, gpt-6-luna/openai-codex, low effort/execution plans. Initial
model responses present; first two completions admitted. Existing transport and
rate-limit exponential retry delays/cooldowns remain. Bonsai/MiniCPM watchers and
GPU ownership unchanged.

Filesystem has only about2GiB free. Added optional --min-free-mib supervisor floor;
new Luna workers use1024MiB and stop before starting another case if below it.
No finish event is written for that unattempted case, so restarting the same queue
and journal resumes it. Default0 preserves the running supervisors' configuration.
State and per-worker journals/results/logs: runs/luna-generation-20260929-two/.


## 2026-09-29 22:17 UTC — storage recovery, eval handoff and primitive evidence audit

The final authenticated1MiB range completed; exact model SHA verified and CPU
tool-call smoke passed. Signed-URL refresh failures now re-enter the downloader
retry loop. Evaluation PID26597 waits for a safe Bonsai boundary; GPU benchmarks
have not started. Preserve Bonsai PID8981 identity until that handoff.

Disk exhaustion interrupted two Luna cases and the first evaluation watcher.
Lossless SHA-checked archival of superseded V5/V6 audit JSONL recovered about896MiB;
archive ledger is runs/luna-generation-20260929-two/storage-archive.jsonl. Resumed
Luna with PIDs26595/26596 and reviewed retry keys only for the two storage failures.
The current filesystem reports42GiB available. Earlier watcher traceback is stale.

New quality finding: primitive WorkflowEvals references immediately return the gold
while initial state rendering can truncate input. Existing directory evidence proof
is insufficient for these references. Do not claim4,194 primitive static conclusions
are fully audited until complete visible state is proved or they are held. Live
teacher IR remains useful. Rebuild on a new frozen runtime, leaving live v2 intact.
Canonical manifest-selected static counts correct the earlier loose-file count:
12,471 native approved decisions/7,910 reducers (63.43%), before this new audit and
final student rendering/dedup.


## 2026-09-30 — automatic full-corpus discovery and persistent carry-forward

User wants all data tracked across pipeline/IR changes. Replaced default omission
of historical native teacher jobs with automatic, immutable admission-filtered
snapshots in both builders. Older static reference sets and source/code lanes remain.
Latest eligible artifact per trajectory ID is selected; older variants are retained
and ledger-linked, never deleted. Overlap with explicit exports is deduplicated
at admission with a note. Default required inputs and previously included artifacts
cannot silently disappear: a replacement/policy decision or explicit input override
is required. Added persistent version/status/next-action catalog and immutable
per-build report, source policy, transformation inputs/outputs and missing-file
retention. Unknown artifacts remain visible migration/review backlog.

Actual full-corpus planning audit:5,490 completed files;2,745 selected trajectories
across2,414 unique programs;684 repeated eligible IDs.3,836 cataloged artifact
paths include derived/archive copies; do not sum them as distinct examples.
Created runs/data-lineage-20260930/recipe.json without executing collection/training.
Pre-joint inventory readiness explicitly blocks Workflow primitive hidden-input
references until repaired/held; generation remains on its existing frozen runtimes.

Raw JSON/source archives expanded the persistent catalog to6,170 artifact paths;
630 are transformation-pending and125 rerender-review under recorded source policies.
These are files (including duplicates/derivatives), not training-example counts.
Readable overview: data/teacher/data-inventory/overview.md.


## 2026-09-30 — complete workflow evidence, admission dispositions and causal DPO

Breaking publication: workflow v2 references must show all source inputs and retain
read/page outputs in the final answer context. Recomputed hash-bound proofs apply
to synthetic references only; actual teachers may select their own evidence. Full
model-free4805-case replay and independent source/oracle/visibility/default-export
comparison passed.28,975 approved native decisions; originals retained, old paths
explicitly replaced in source policy and unsupported old references fail closed.
32k reference replay context replaces16k after37 compaction-induced evidence holds;
student limits remain enforced later. New read turns change denominator: four newer
bundles32,993 decisions/7,910 reducers~23.97%, so earlier63.43% report is obsolete.
Target remains25% unique admitted final train decisions; expand legitimate supply.

Raw exclusion reasons now carry migration/replay versus source/oracle review versus
candidate failure dispositions. Held-out data remain evaluation-only; no automatic
DPO-negative label from an exclusion. Snapshot policy/runtime identities recorded.
Offline correction/preference recipe stages write lineage and candidate audits.
43 causal /2 preference pairs recovered conservatively (35 failed_action,8 wrong_result),
source groups retained. Five old /1 pairs remain backlog. Student rendering/split/
token/dedup review still pending; no training started.

MiniCPM GPU pilot returned durable11 completed cases,4 accepted/24 planned. Collector
aggregate export had been overwritten per case, causing erroneous zero-collected
report. New recovery utility reassembles saved student/test jobs, checks exact IR via
native scorer and records source hashes; original report retained. Local evaluator
now exports per-case then aggregates. Remaining13 cases missing/timeouts/not reached;
application probes not reached before30-minute limit. Bonsai restored cap4/runtime-v39
PID35895 with fresh replies. No new GPU reservation launched for this accounting fix.

Prior Luna pilot finished; authorized two-worker generation continues on fresh32-case
all-directory Workflow batch, source families interleaved/shorter inputs first,
no duplicate prior completed IDs. PIDs36471/36472, frozen visible runtime-v4,
cap2/noGPU, backoff/cooldown/diskfloor retained, unique per-case output paths.

Offline migration pilot:10 short accepted native teacher traces held only for obsolete
outcomes. Initial draft produced2 same-answer review candidates but recorded changed
observations; parent tightened the migrator to reject any changed observation.
Strict-v3 yields0 candidates:2 changed observations,8 unmatched/ambiguous original
call ownership. No training promotion. Added admission-level hold for any history
migration or migration_candidate row, so even explicit builder inputs cannot bypass
review. Pilot artifacts and originals retained; catalog tracks review-candidates as
transformation-pending. Future changed-context migration needs a separately reviewed
provenance/evidence contract; no automatic relabeling of these holds as model failures.


## 2026-09-30 — JSON file formatting is not a semantic failure

User explicitly requested fixing whitespace-only rejection. Default folder checks
now structurally compare JSON content with strict field/type/array/string preservation,
invalid/duplicate-key/unsafe-number rejection and exact file-set coverage. Other text
remains exact; explicit compare exact is available for byte-sensitive fixtures.
No global whitespace stripping or partial-credit tolerance. New provenance/reuse
identity json-content/1 prevents old strict-verdict reuse, while existing stronger
positive exact evidence remains valid (no broad data-quality-version bump).
Built Node and froze new runtime for future campaigns; active frozen workers unchanged.
Actual saved MiniCPM file rescore changes only web release (newline) to accepted:
5/24 planned vs4/24 original. Original reports/results remain; no new GPU/provider
calls. Full failure analysis records semantic errors, recovery loops, false blocked
stops and active long-generation timeouts; partial tool outcomes remain unverified.
Held-out traces stay out of training; use new independent training variants for patterns.


## 2026-09-30 — stronger directory output contracts and continuing hourly checks

Luna directory batch32 completed with28 accepted: one envelope instead of mapping,
one pair of string booleans, two semantic judgment mistakes. Generic Record<string,
unknown> let format mistakes end the call rather than guide repair. New directory-v2
schema fixes each named output field to the type derived from source questions,
not their gold answers. New adapter revision visible-inputs-typed-batches-v3 and case
IDs; source IDs/groups/labels preserved. Do not republish or overwrite static v2 merely
to incorporate typing. Native parser previously rejected quoted field names; fixed
parser/formatter/compiler target conversion and JSON escape handling so source hash
keys can be typed. Existing duplicate-field/type checks remain. Eight fresh native
references passed with complete-input proofs and no failed actions/model calls.

Two authorized Luna workers39580/39581 continue on64 fresh directory cases with typed
contracts, runtime-v3, per-case exports, cap2/backoff/diskfloor retained. Native replay
pilots stay review artifacts. Bonsai stays cap4; exact-PID boundary monitor39325 waits
for current case then adopts JSON-content runtime-v2. Frozen runtime files/GPU server
are unchanged. Explicit user request resumes hourly sleep/check/rejection investigation;
check/log/handover updates continue. No student training or automatic DPO negatives.


## 2026-09-30 — callback contracts, readable jobs, and reserved evaluation lineage

Bonsai relational-policy125/133 exposed an actual compiler bug: unannotated async
callbacks discarded their caller-declared Promise<boolean> target for nested nl.
False-bearing objects could consequently pass JavaScript truthiness checks. Contextual
caller signatures now determine the awaited target; predicate prompt example uses
nl<boolean>. Native build passed; saved failing-shape compiler audit yields boolean
and no diagnostics. No broad conversion of objects to booleans or weaker checks.

Typed Luna batch64 finished61 accepted/3 source-verified judgment errors, no output
format errors. Two final answers inverted the model's own conclusion, one misordered
explicit card-lock authorization. Readable node/question keys plus final per-field
polarity check replace opaque hashes in fresh directory-v3 cases. All original instance
IDs, labels and groups retained; canonical static-v2 publication unchanged. Two workers
on499 fresh cases;8 actual reference replays passed complete-input proof, no provider calls.

**Breaking decision:** references.mjs reserves s102 probe and s900 test; no native
training release exists. Historical generation incorrectly labeled s102 train. Audit
found1,835 unique accepted historical teacher/train trajectories,1,575 previously
admitted. New central hold held_out_reserved_curriculum excludes them, retains source
artifacts/ledger and routes to evaluation rather than model failure. Generator rejects
reserved train seeds; old materialized SFT/stages and DPO have downstream guards.
Refreshed snapshot selects1,278 trajectories/1,121 programs and tracks2,775 raw reserved
artifact holds. Old snapshots remain; regenerate admission/stages before training.
Bonsai queue-v32 preserves the active case, holds56 future reserved roots and retains18
clean mixed-batch roots. Boundary runtime rollout keeps cap4 and no worker overlap.

TATQA128 is source ambiguity: chairman age67 is explicitly Executive Chairman of the
Board of Directors. Gold averages four exact Director rows (58.75); including chairman
produces60.4. Added source review hold without relabeling either interpretation as a
model negative. Hold propagates to current generation/admission; frozen workers retained.
No GPU evaluation rerun, unit tests, student training or automatic DPO negatives.


## 2026-09-30 02:08 hourly check — explicit field maps and complete eval pages

Both Luna workers progressed128 cases/120 oracle-accepted by check start; no transport
failures/timeouts. Audit eight: five misread fixed schemas as incompatible with filename
maps; one large repeated retrieval ended in any(...) ReferenceError and false block;
one genuine claims-supported error; one rubric ambiguity. Directory-v4 root prompts
explicitly enumerate the fields, identify them as filename stems and explain placement
inside return_result.value. Source labels and inputs remain unchanged. Eight real
reference replays passed complete-input proof;7 reviewed retries ran on the new prompt,
all6 schema/retrieval blocks repaired. Case77 still misclassifies directly tool-supported
timelines using a separate quoted agent policy; retain the real failure, no special
answer override. No preference pairs inferred across changed prompts.

Structured eval previews now offer read_page starting at page1 for the full value.
Previously the full value existed in transcript but repetitive evals reproduced the
same structural cutoff. Native large-input source38 replay read10 full pages and
completed; audit-only modified scripted trace is not training. Reference page follower
recognizes the new hint. Node build passed; active runs moved at journaled boundaries:
Luna runtime-v5/cap2; Bonsai runtime-v6/queue-v33/cap4. No GPU server change.

Bonsai long FOLIO case148 performed real delegated work then rejected; source formal
premise uses exclusive conservative-or-Republican, while English either/or conclusions
are ambiguous and no conclusion FOL is released. Exclusive readings can support the
source labels. Added whole-story395 review hold, preserved gold. Workflow claim435c...
bank-advice/factual-assertion ambiguity also held. Three future source-held Bonsai roots
removed while retaining current case and other roots; case150 completed accepted.

**Publication/course decision:** source-backed/recovered pass current admission;
TATQA hold blocks one directory-expansion static case, Workflow hold blocks one
primitive and one batch. Instead of leaving both whole bundles unavailable, publish
quality-filtered revisions preserving all original files/golds/trajectories. Generic
filter-static-bundle.mjs hash-checks IR/result pairs, only permits explicit source-review
or reserved-evaluation holds, rematerializes flags, validates retained admission and
publishes manifest last. Workflow quality-v3:4,803 cases/28,958 decisions (2 held);
directory quality-v3b:1,305 cases/3,529 approved decisions (1 held), preserving1,030
existing disapproved decisions. First directory filter assumed all decisions approved
and aborted; scratch quality-v3 was never published. All4 bundles32,969 approved
source decisions. Explicit hold ledgers exclude source disputes from DPO negatives.
Replacement chains now resolve transitively with cycle rejection, preventing a later
quality revision from looking like lost older data. Added corpus-policy dependency to
stage preparation so resume cannot reuse splits after a policy change.

Planning recipe-v6 now includes current static publications and automatic snapshot
(1,448 trajectories/1,291 programs) with no inventory omissions/quality blockers.
Final student/template/token/group/dedup/25%-mix audits still pending; no training run.


## 2026-09-30 03:56 UTC — equivalent answers, finite loops, balanced Luna queues

Root source/rubric review overrides two overly broad audit suggestions: Workflow135's
handoff_required asks policy/entitlement escalation, not missing prerequisites or an
actual voluntary escalation; retain model-error classification. TATQA176 has correct
103.1 but no evidence for invented million; no annotation hold solely for conventional
unit inference. A separate scale-visibility audit precedes global prompt changes.
claims_supported accepts factual evidence from tool error payloads; success is needed
for claiming an action completed, not for every fact in an error response.

**Admission decision:** MuSiQue answer mismatches now receive the existing extractive
answer-equivalence review reason. Two current rejects are supported paraphrases
(North Korea and China; Western Europe). Preserve original labels/verdicts and exclude
unreviewed negatives/handoffs/materialization. Exact accepted positives remain useful;
no fuzzy global text oracle, source-label rewrite or wholesale generation pause.
Five actual collected rows audited under rebuilt admission; two exact positives pass,
two paraphrases held, TATQA invented scale remains wrong_return. Alias-based repair
still requires explicit reviewed source contracts and faithful replay.

**Compiler usability:** const chunk=3000; i+=chunk is finite but old literal-only
counter guard caused22 repeated refusals in one saved trace. Permit only a preceding
same-block/source const with finite positive direct NumericLiteral. No alias/expression
or outer-scope evaluation. Reject counter-name shadowing, mutable/dynamic/zero/negative
steps; existing bound/body guards remain. Saved snippet now compiles; comparison audit
preserves refusals and literal success. Build passed; no unit tests. Collector adds
counter_loop_policy_version2 to provenance/reuse keys. Frozen runtime-v7 prepared;
Bonsai monitor62418 moves54068 only after its journaled case boundary. Luna staysv5.

**Queue operations:** static even/odd Luna assignment left fast worker nearly done and
slow worker about150 cases behind. Added coordinated stop barrier/old_journal/history
carry-forward support. Both old supervisors53036/53160 completed current roots and
stopped before new61176/61177 launched.150 pending entries split75/75 by estimated
input size, source/index/seed/jobs preserved. Worker-specific new outputs prevent races
when moved entries originally shared old output paths. Both distinct journals retain
hashed old histories for resume; copied finishes are not additive progress. Actual
rollover confirms no overlapping workers. Authority balanced state; cap2/noGPU/backoff.

At03:52 raw chain361completed/342oracle-accepted includes retries, not unique admission.
Three fresh rejects are two outcome-judgment mistakes and one incomplete-reading block;
no new source holds/schema/runtime errors. Recipe-v7 planning snapshot1,638selected
trajectories/1,481programs, canonical static decisions32,969, no included blockers or
missing defaults. Student rendering/token/group/dedup/final25%-mix still pending.


## 2026-09-30 04:07 UTC — source unit visibility and explicit filtered carry-forward

Luna's254-case TATQA audit found209 nonempty golds with visible matching unit cues,
two without explicit scale, and one empty gold conflicting with explicit target units.
Root read all three full tables/notes and confirmed: unbilled receivables13026d09...
million notrecoverable; zero cashdeposit difference36308f38... thousand notrecoverable;
netcash difference2b89071f...376.2 has explicit millions but blankgold. Added narrow
source holds, preserving annotations/evidence and excluding scale-only DPO labels.
The earlier103.1weightedshares case has no unit and blankgold agrees, so model-invented
million remains a real unsupported inference. No global missing-unit default yet.

Published quality-v4 directory static after full retained admission/hash verification:
1,302cases/3,495approved decisions,1,030existingexcluded decisions;3new holds/4cumulative.
Prior quality-v3b manifest/files retained; catalog3b→4 chain. All4static32,935approved.
Recipe-v8 no omissions/blockers; snapshot1,676trajectories/1,519programs, planningonly.
Queue-v34 removes3futureheld TATQAroots while retaining9clean mixed-batch members/current.
Frozenv8 includes holds/loop fix. V8monitor65189 waits for v7monitor62418 rollout, checks
expected predecessor queue, obtains exact new PID, then waits another journaledboundary.
No active queue/runtime mutation or overlapping supervisors. Helper predecessor wait is
bounded2hours and fails without signaling if incomplete. New audits/originalfiles retained.


## 2026-09-30 04:13 UTC — reviewed MuSiQue aliases and independent fresh variants

Two exact source/evidence-scoped oracle allowances: source4hop1__205937_144938_83779_69861
accepts North Korea and China alongside original North Korean-Chinese forces; source
3hop1__478606_751065_78953 accepts Western Europe alongside original north-westerncoast
answer. Both are explicitly supported by supplied articles and suitable for the actual
question. Registry matches exact primary gold, original snapshot and full evidencefile
hashes/text; drift fails closed. No general token overlap/fuzzy matching. Futureadapter
and historicalIRclones share applyReviewedMusiqueOracleAlias, distinct :reviewed-alias-v1
IDs and reviewprovenance; visibleprompt, inputs, primarygold/reference preserved.
Four actual native audit replays accepted bothprimary andalias returns. Audit records
are not training; raw failed original trajectories remain unchanged/reviewheld.
Queue-v35 prepends2 fresh independent Bonsai variants with newseeds/jobs,2400sec limits
and existingcap4, then preserves filteredv34work. V8monitor65189 still awaitedv7 and had
not initialized/signaled a worker: stopped that exactpending monitor, preparedv35,
restarted66207. No live queue mutation. V7monitor62418 still owns currentboundary;
66207 waits predecessorcompletion, derivesexactnewPID, then uses nextboundary.


## 2026-09-30 04:21 UTC — generic evidence-scale prompt revision

After holding contradictory/missing-unit source contracts, future TATQA roots now
clarify that scale follows the question/source evidence for the requested quantity,
empty for dimensionless or unstated units, rather than financial conventions/unrelated
rows. Shared applyEvidenceScaleContract produces distinct :evidence-scale-v2 IDs and
revision metadata; source golds, references, source groups and evidence stay unchanged.
Idempotence and structural preservation checked; saved failed176 variant native replay
accepted103.1/empty scale in5turns. Audit-only replay is not training; a new independently
sampled teacher retry isqueued instead. No specific gold hints or pairs across prompts.
Queue-v36 adds this1fresh retry ahead of2reviewedMuSiQue variants/filteredv34work. V8
monitor66207 was still waiting for v7, with no rollout state/no worker signals; stopped
that exact pending watcher before creatingv36, restarted67297. ActualBonsai54068/v6
currentcase remains immutable; v7monitor62418 thenv8monitor67297 own serialboundaries.


## 2026-09-30 04:41 UTC — continuing two Luna workers on audited file/tree reducers

Prepared559source-backed roots from canonicaldirectory quality-v4:287CommitPack,
251TATQA,21TreeDST. No quarantine/generation-held cases; sourcegold/evidence/group
lineage preserved. TATQAuses newevidence-scale-v2 contract. Sixnative references match
expected/accept/admit. CommitPack all287exact-file oracles audited againstvisible
replacement requests, unique target/span, sourcefiles preserved: no hiddeneditgoal.
Cost-balanced new279/280roots. Newqueues retain all old balancedkeys so finishedskip
and remainingWorkflowcases complete first, avoiding lostwork/redundantgeneration.
Coordinatedboundary rollout69128 moved old61176/61177 to69180/69181 on frozenv8;
both oldsupervisors/children stopped before either replacement. Existingbalancedjournals
retained, including duplicatedcarry-forward histories; sourcejobdirectoriesunchanged.
Authority newfile/treecampaignstate; cap2/noGPU/backoff unchanged. This continues
file/treecoverage toward25%final unique admitted decisions, not a rawrow targetclaim.
Bonsai largeFOLIO212 remainsbounded3600sec, manyfreshreplies and repeatedwholebatch
work; read-onlyLuna audit considers schema/retry/lifecycle causes, no prematureclaim
that freshdecode is usefulprogress. V7thenV8 serialboundarymonitors remainauthority.


## 2026-09-30 05:04 UTC — scoped child lifetime and useful partial telemetry

Slow FOLIO212 timed out at3610sec after109fresh replies/72623completiontokens.
Old partials lack tool observations/call IDs, so exact orphan-response counts cannot
be recovered. Code inspection found a real amplification hazard: Promise.all first
rejection let sibling NL calls survive while the parent started another batch.
Track children by caller ID; drain the current invocation's cohort with allSettled
before eval releases locals or invocation closes its environment. Nested children
settle transitively, original errors preserved; no global drain or blanket cancellation.
BREAKING: successful eval with still-pending unawaited NL calls now settles them then
returns a clear await-all error before snapshot/coercion/commit. External child effects
are not claimed rolled back. Runtime contract18 prevents old checkpoint reuse under
new semantics. Existing admitted positives are not blanket held by version alone.
Node build passed; actual source referenceRow172/182/212 parallel batches accepted
(8/5/8turns). Audit summaries only; no training rows/provider/GPU calls/unit tests.
Old failed partial cannot faithfully replay, so failure-path verification is limited.
Partials now add invocation ID, request/observation timestamps and bounded2000-char
last tool output preview/hash, without changing request hashes or model context.

05:00 full inspection: Bonsai70991 on frozenv8/queue36, cap4/server4/GPU99%; both
serial v7/v8 rollouts completed. Luna69180/69181 on frozenv8, two provider workers
continue new559file/tree roots after priorWorkflowkeys.96completed/69accepted/27
rejected at inspection; raw run counts, not final admitted/deduplicated mix. Generic
TATQA scale retry accepted103.1/blank; reviewed western-Europe retry accepted;
NorthKorea-and-China question returned only NorthKorea and still failed (no broad
relaxation). Current source/FOLIO425 wording and fresh Luna failures under review.


## 2026-09-30 05:09 UTC — narrow FOLIO425 hold and runtime-v9 boundary rollout

Independent sourceaudit confirmed pinnedFOLIO425 HaveCars→Drive abstraction absent
from visible chooses-to-drive wording; explicit choice/action countermodels invalidate
English-only certainty onC3-C6. Central pending sourcehold includes historical variants
and prevents falseDPO negatives; golds preserved, no globalFOLIO interpretation change.
All4canonicalstaticbundles containzero425rows, so no static republish. Root audited
actual queue ranges:6future single roots held pluscurrent425root preserved untilboundary.
Queue37 preserves361otherentries; originalqueue36 retained. Agent initial explicitkey/
limitedrange count missed these roots, superseded by root queue.audit.json.
Frozenv9 includes childdrain/telemetry/contract18/sourcehold. Exact-PID boundarymonitor
74250 owns Luna69180/69181 andBonsai70991 independently, samejournals/caps; Bonsai
nextqueue37. No in-flight runtime/queue mutation. Initial shell background launch did
not survive/initialize; verified no monitorstate, then detachedPopen74250 started.

Recipe-v9 refresh completed: 1851 selected generated trajectories/1674 programs, no missing default inputs/included quality blockers; static approved decisions remain32935. Planning snapshot, not final training export.
Luna runtime-v9 boundary rollout completed: worker1 PID74281, worker2 PID74312; authority/check updated. Bonsai70991 still awaiting currentcase boundary under monitor74250.


## 2026-09-30 05:16 UTC — three narrow TATQA aggregation/change source holds

Independent review and root input verification establish ambiguous question contracts:
e27c8621... total at two dates can mean two net balances versus sum; f6ef3a62...
all years as of2019 can mean all reporting columns versus2019-only;7a6c059d...
percentage change in net-sales share permits relativegrowth versus percentagepoints.
Preserve allgolds, centralpendingholds exclude generated/static/DPOoutputs. Allthree
newLunacampaignindices311/404/416 alreadycompleted, so no activeLunaqueueedit.
Agent report initially included another question's table in IndustrialSolutionssection;
root read actualtable(30/28shares,net-salesnote), requested report correction before
policy publication. Do not rely on copied summaries without source verification.
Directoryquality-v5 filters only3cases after fullIR/results hash/admission checks:
1299cases/3459approved decisions;36decisions removed,7cumulativeheldcases. Original
quality-v4files/golds retained, catalog4→5replacementchain. Fourstatic32899approved.
Proportion0.14 versus13.968percent is representation-equivalent, not arithmeticerror;
no sourcehold or broadfraction/percent admission change without reviewed formatcontract.
TATQA numericrepresentation/precision improvements under focusedreview; rawnegatives
remain preserved for audit and should not be assumed causal DPOpairs.


## 2026-09-30 05:29 UTC — opt-in TATQA numeric answer contract

New tatqa-answer-record oracle/files comparator validates strict exactly answer/scale
strings, duplicate-key rejection, exact enumscale, standalone decimal sign/grouped
commas and decimalBigInt canonicalization. Numerically equal displays(+6/6,12.0/12)
and unambiguous two-decimal rounding(21.7674/21.77,2.5671/2.57) agree. Half-cent
ties retain exact equality only to avoid falsely claiming Pythonfloat parity. Numeric
strings bounded512chars; no extracting numbers from prose/currency/%/unit text or
scale conversion. GenericJSON/TreeDST policies unchanged. Returnedrecord must agree
with answer.json under same comparator; sourcefiles remain exact/preserved.
Source official evaluator rounds numericanswers to2dp; upstream scale_em is a separate
metric, not a mandatory EM gate. Our exactscale requirement is deliberately stricter.
Source: https://github.com/NExTplusplus/TAT-QA/blob/master/tatqa_metric.py .
Future numeric-only TATQA variants add :numeric-answer-v1 after evidence-scale-v2,
explicit visible numeric-display instructions, oracle/files normalization metadata.
No hidden annotatedscale instruction: root caught/removed it before freezing. Helpers
composeidempotently. Originalgold/evidence/groups/reference preserved. Native4reference
audits accepted; actual4oldcandidate oracle/files/returnagreement checks pass. Their
raw historical verdicts/bytes remain unchanged and central legacy_tatqa_numeric_display
_oracle hold prevents falseDPOnegatives pending migration review; no auto promotion.

Prepared futurequeuevariants:559Lunacases,249numericIRupdates;1229BonsaisourceIRrows,
255prompt/oracleupdates(includeevidence-scale revisions). Three sourceheldLunakeys
omitted(alreadycompleted). Bonsaiqueue38 removes3futureheldTATQAroots and retains
6cleanmembers from2mixed4-rootbatches withdistinctsplitkeys; currentheldFOLIO425
root preserved untilnormalboundary. Joboutput directories, seeds/caps/history retained.
Manifest/hashes/preservationaudit hourly/tatqa-numeric-campaign. No in-flight source
orqueue edits; original sources stillon disk. TwoLuna boundarymonitor77277 owns current
74281/74312→frozenv10/newqueues. Bonsaimonitor77278 waits v9monitor74250state, derives
exactnewPID/queue37 then nextboundary→v10/queue38. No overlappingcollectors/restarts.
Recipe-v10 refresh started after threeholds/currentpolicy; no studenttraining/tests.

Luna-v10 rollout completed:77324/77384, newqueues/frozenv10; authorityupdated.
First7newnumericcasesaccepted (small sample, not comparison-rateclaim). All249
convertedLunanumericroots have original-arithmetic-gold adaptation (not year/count
spans). Recipe-v10:1905selectedtrajectories/1704programs, no includedblockers/missing
defaultinputs, fourstatic32899decisions. Systematic TATQAoperationwording audit ongoing.

05:34UTC log correction: matched recent note headings to actual commit/artifact
UTCtimes (numericfreeze05:27, rollout/snapshot05:28, commit05:29/latestnote05:32;
three-holdpublicationcommit05:16). Earlier estimated05:37/05:43headings ran ahead
of clock; corrected. Authoritative processevents retain actualepochtimestamps.


## 2026-09-30 05:43 UTC — quantities need their own units, not table-wide cue matching

Newnumericrejection c9026c61... employeeFTE421.1−413.9=7.2, goldmillion, modelblank.
Root read fullmixedUSDmstaff-cost table, separateemployee-countsection,108seafarer
note andFTEdefinition. Million appliesmonetary costs, notpeople; centralpendinghold
retainsgoldandblocksfalseDPO. Earlier unitvisibilitymatchingcannot certify that a
unitbelongs to requestedquantity. Independent quantityattachmentaudit inprogress.
Rootalsoverified8b50fee5... sourceprovidesSanmina shares17.7/16.0%, questionasks
revenuechangewithnoamounts/annualtotals; holdpendingexplicitshare/point operation.
bd73aadd... asksbalance-sheettotalaccruals2019/2018, goldcombined541+567 whereas
individualas-ofvaluesaresupported; narrowholdmatchingearliertwo-dateassetcase.
Milderannualconsolidatedrevenuecandidatec35adb... notautomaticallyheld; totalrevenue
in2018and2019 reasonablyspecifiescombinedsum, no additional observeddisputeyet.
Other namedshare-changequestionswithexplicitmetriccontextnotheldbykeywordalone.
Directoryquality-v6:1296cases/3439approved,20decisionsexcluded,10cumulativeheldcases;
fourstatic32879approved, priorv5retained/catalog5→6. Newsourceholdpatchbuilt butnot
frozen/rolleduntilquantityauditdone; do notmutatev10 orpendingv9/v10Bonsaimonitors.
Nextsnapshotneededafterholdbatch. Currenteligiblefile/tree553cases(287CommitPack,
245TATQA,21TreeDST) after6newTATQAholds; original559sourcecasesremainpreserved.
TreeDSTcaseonlyChineseFood/Chinesefood differenceaudited; sourceexactmatchmetric
isstrict andutteranceusesChinesefood, so no globalleafcasefold basedononefailure.
Structuralname/collisionsemanticsandnumeric/nonlinguisticvaluesmustremainexact.


## 2026-09-30 05:47 UTC — consolidate pending Bonsai rollouts

Quantityunit auditindependentlyconfirmsemployeeunitdefect andfiveothershareunits
supported. Noextra blanketunit holds. Frozenv11finalholds; newLunaqueuesomit3latest
heldkeys, Bonsaiqueue39 removes3futureheldroots/preserves6cleanmixedbatchmembers.
COURSE CHANGE: stop obsoletependingBonsaiv10watcher77278 thenv9watcher74250, after
exactcommand/statechecks(v9Bonsaiwaiting/no newPID, v10nostate/predecessorwait).
Teacher70991/v8currentcase andserverunchanged; alreadyadvancedLunateachersuntouched.
Solev11monitor79671 nowownsall3currentboundaries, Bonsaidirectv8→v11/queue39,
Lunav10→v11/latestfilteredqueues. Avoid2intermediateBonsairestarts/casewaits.
Oldmonitorstatesretainedassupersededhistory, proof runtime-v11-supersession.audit.json.
No journalfinishfabrication or interrupted modelwork; currentboundedheldroot finishes
normalbudget andrawpartialpreserved. Newsource/queuefilesareimmutableindependent
variants; samecaps/seeds/jobs/journals. Luna1new79716, Luna2pending77384 atinspection.
Recipe-v11refreshstartedafterlatestholds; planningonly, no training/eval/tests.

05:55UTC allthreev11rolloutscomplete: Bonsai80478directfrom70991/v8toqueue39/v11;
Luna79716/79801v11 latestfilteredqueues. HeldFOLIOcurrentrootnormalboundedtimeout
finishpreserved; nexteligibleBonsaicasefreshresponses/GPU93%. NointermediateBonsai
v9/v10restarts. Latestrecipe-v11:1964trajectories/1740programs, static32879decisions.
Newnumericrejectionauditongoing; nextfullhourcheck~06:04.


## 2026-09-30 06:08 UTC — full hourly check and next reviewed unit repair

Allteachersv11:79716/79801Luna,80478Bonsai queue39; freshprogress/GPU98%,disk23.31GiB.
Newfile/tree319completed uniqueorigins:CommitPack122accepted/1reject;TATQA124/51;
TreeDST19/2. Numeric72completed/63rawaccepted/9reject. Counts are attempts, not
finaladmission; currentworkingnewholds supersede pinnedv11candidatepolicy. Nextcheck
07:08UTC incheck.json. Currentpartials nowreallystore invocationID/time/toolpreview
andhash(bounded2000chars), verifiedactualBonsairuntime18metadata, notmerelycode.

Rootfullinputreviewconfirmed87fe0836... displayed20.5percentchange versus15.8million
goldwithoutspecifiedoperation;81304522... changebetweenvestingcategorieshasno
subtractiondirection;d3a5439d... unitlakhcan'tbeexpressedbyourenum/newpromptwhile
goldblank. Threecentralpendingholds; directoryquality-v7 removes23decisions/3cases:
1293cases/3416approved,13cumulativeholds; allfourstatic32856approved. Originalgolds/
traces/publications retained, catalog6→7. Runtimepatchnotyetfrozen pendingunitrepair.
Proportion2f301...97.21/empty versus0.97/empty remainsreal100x math/formatfailure,
no toolerrors. Removedstaleadmissioncommentclaimingalldelegateditemsrequired; actual
logicallowsdirect/mixed/delegatedcorrectsolutions, unchanged.

SYSTEM FORMAT BUG: Generic evidence-scale/numericprompt made unlistedlakh units
impossible despite original242.5/blank sourceconvention. Do notblamehonestblocking
model. Lunaadmissionagentpreparingexactsource/evidence-scoped visibleunitcontract
variant:explicitnumericunitsstaylakh,scaleempty; preservegold/ref/evidence/groups,
distinctID andstrictreviewvalidator. Oldblocked/raw/sourcecaseheld; no automatic
positivetransform. Futureteacherfreshretry/referenceaudit onlyafterrootreview.
No blanketunitconversions, no unsupportedgoldrewrites. Unitcontractapprovalmust
beexplicitvisible andproofbound; centralholdexceptiononlyforreviewednewvariant.

## 2026-09-30 06:25 UTC — reviewed lakh representation and runtime-v12

- System format defect: the generic TaTQA prompt demanded a source unit but excluded
  lakh from the scale enum. A distinct, source-specific visible prompt now explicitly
  retains the displayed lakh quantity with empty scale. Original 242.5 gold, table,
  notes, reference, source groups and hashes remain unchanged.
- Admission exception applies only to the exact reviewed variant with immutable
  identity/prompt/gold/input/reference checks. The original blocked row stays held;
  no failure becomes a positive or DPO negative through reinterpretation.
- Native reference replay passed all checks in five turns, with unchanged inputs and
  matching answer.json/return; build passed. No tests or model calls in this audit.
- Frozen runtime-v12 deployed to all three teachers at journaled case boundaries.
  Luna PIDs 84926/85014; Bonsai 85050. Provider/GPU caps unchanged (2/4). New queues
  exclude future held source cases while preserving historical/current attempts.
- Fresh independent Luna retries: reviewed lakh variant and the real CommitPack
  README truncation failure. Distinct jobs/seeds/keys preserve both original failures.
  These are not automatically causal DPO pairs. Preservation manifest and rollout
  evidence live under runs/generation-check-20260930-hourly.

## 2026-09-30 06:34 UTC — numeric delta source quality holds

Independent audit of five new numeric rejects, then root review of full source
inputs and references: four source/prompt issues and one genuine row-selection
model error. Central holds now cover expense sign convention cc42e86a, wrong-column
gold122bddf9, absent magnitude unit3257598a, and year/average ambiguity85d145d7.
Gold122bdd uses Other Accounts for both years, not mixed columns; its question names
Costs and Expenses. Preserve golds and all originals; do not promote model outputs
or use these as DPO negatives. Published static quality-v8 excludes three affected
canonical cases (1290 cases,3397 approved decisions); one held source was absent.
Frozenv13 boundary rollout retains current attempts/history/clean batch members,
with unchanged worker caps. Final student audits remain pending.

## 2026-09-30 06:42 UTC — correction of audited source identity

The cc42 expense audit UUID had c56c where the saved source is c56d. Root copied
that typo into the hold; an independent policy count/artifact join and the broader
Luna sweep caught it. Corrected exact source ID, mechanically checked all five
reported identities against saved IR, and published quality-v9 excluding the now
matched case. Contrary to the earlier log, the fourth source WAS in canonical
static data; it failed the identity match. Preserve original audit evidence/backup,
raw source/gold/traces and all prior static publications. No labels or DPO pairs
changed. Frozenv14 rollout applies the corrected hold at teacher boundaries.

## 2026-09-30 06:49 UTC — artifact identity guard for source reviews

Added a reusable read-only source-audit identity verifier. It validates exact reported
source ID, optional program ID/question, and saved external-source identity against
the artifact IR; pins hashes and writes new immutable audit evidence. Actual 0625
report reproduced one mismatch, with other four cases matching. This is an audit
workflow guard, not an oracle, admission relaxation, unit test, or training promotion.
Apply it before copying reviewed IDs into policy. Corrections retain original reports.

## 2026-09-30 07:00 UTC — numerical target correction scope clarified by user

User permits considering bounded injection of an exact training target for numbers
that represent model judgment. Computations/code outputs based on exact inputs
must remain exact. Existing explicitly stated rounding/display equivalence remains
valid. Preserve raw model outputs and identify any injected example as a synthetic
correction with target provenance, bound/rubric and consistent surrounding trace;
never pretend the original model predicted that value. No correction is active yet.
Current source audit concerns table arithmetic/units/question scope. File/edit counts
and sums following semantic classification must match the actual decisions/actions;
these are not free numerical ratings eligible for terminal-value substitution.
No active subjective numeric-rating mismatch was found in the inspected generators;
workflow confidence is source-label admission metadata, not a numerical output
that needs target injection. Source/oracle conflicts remain review holds.

## 2026-09-30 07:20 UTC — complete source sweep and next Luna assignment

Root checked all16 conservative TATQA review findings from the239-row sweep before
central holds/staticpublication. Scope, units, period/direction, population and
prose-versus-rounded-table conflicts are held without rewriting golds. Unsupported
agent recommendations were withdrawn after full-input review; retain correction
history. EPS direction and goodwill population are ambiguities, not proven arithmetic
errors. Staticquality-v12 keeps1273 cases/3276 approved decisions (allfour32716), with
explicit transitive replacements and prior artifacts preserved.

Luna original queues exhausted normally. Prepared304 new non-TATQA directory cases
(248MuSiQue/56SciFact), balanced152 each, are appended in new immutable queues with
same journals/two provider workers.301 matching future Bonsai tasks are assigned to
Luna to broaden generation without redundant future GPU work; preserve current and
completed attempts. Other prior failures remain explicit repair backlog, not complete
training data. QASPER generation remains held; its independently verified statics
stay included. Runtime-v15 was never deployed; exact-PID boundary watcher93792 rolls
v14 directly to frozenv16, maintaining caps4Bonsai/2Luna and backoff.

Refresh immutable full training snapshots after consolidated audit batches/hourly
checks; their6.2GiB cumulative footprint makes per-small-change refresh wasteful.
Current policy/lineage holds are published immediately; clearly mark prior recipes
as stale until refreshed. No deletion of preserved data or student training.

## 2026-09-30 07:32 UTC — exact computations and current inventory

User explicitly confirmed code/computational results from exact inputs must be exact. Bounded synthetic numerical correction remains only a possible future policy for subjective judgments, with original output/provenance retained; no injection enabled. Recipe-v16 refresh includes current source holds, four static bundles/32716 approved decisions and2306 selected generated trajectories/1881 programs; no missing defaults or included quality blockers. Final student formatting/split/token/dedup/25%mix audits remain required.

Root read MuSiQue2hop__128979_90736 and SciFactclaim347 saved inputs. MuSiQue full date June19,1985 is explicitly supported by same article behind year1985; prepare strict pinned alias/fresh independent retry, do not promote historical trace. SciFact supplied insulin-maintains-progenitors sentence supports the differentiation claim by causal inference; retain SUPPORT. Absence of an explicit suppression experiment does not establish CONTRADICT. External full-paper facts are not added to visible evidence. Evidence: hourly/rejection-audit-0708-other/root-adjudications-0732.json.

## 2026-09-30 07:45 UTC — source-specific monitoring and scoped Markdown edits

Fixed queue supervisor index collisions: next Luna worker2:0012 falsely inactivity-timed-out after300seconds while exact partial had45 saved replies; old glob selected unrelated original index164 result with5. Resolve sourceIR with frozen collector recordDigest once per entry; require programID/provenance digest; choose latest matching partial/result once. Scope retry deadlines likewise. Preserve raw timeout/evidence; independent retry under new key, no model-negative DPO label. Budgets/backoff unchanged.

Versioned opt-in Markdown EOF comparator permits one final LF/CRLF variation in changed allowlisted existing .md target only. Body/codefiles/unchangedfiles exact; missing targets, unexpectedfiles, extra blanklines fail. CommitPack adapter proves unique nonempty replacement with exact gold; EOF-only tasks retain exact original. Prompt says replaceonce, preserve unmatchedbytes, do not repeat if find remains inside replacement. Native variants73/136/180 pass; old136/180 migrationheld, duplicate73/truncated42 stillfail. No historicaltracepromotion. Third pinned MuSiQue alias accepts supported June19,1985 precision of original1985 gold; source/gold/reference unchanged.

All teachers now frozenv17 after journaled boundaries: Luna97137/97063, Bonsai97277. Same two Luna workers/providercap2, Bonsai/GPUcap4, server/journals. Five independent reviewed/infrastructure retries precede continuation. Recipe-v16 predates classifierchange; one hourly refresh next. Evidence: hourly/supervisor-program-identity/audit.json, commitpack-markdown-edit-contract/current-v17-audit.json and reviewed-musique-date-alias. Build/nativeactual/saved-input audits only; no tests/studenttraining.

## 2026-09-30 08:05 UTC — exact numeric comparison and six source holds

Generic normalized numeric strings formerly used JavaScriptNumber, allowing distinct largeintegers/highprecisiondecimals/overflow values to compare equal. Replaced with exact decimal canonicalization up to512chars (longer values fall back to rawtext), preserving numeric formatting equivalence without generic rounding. QA span numeric guard compares exact identities, tokenF1 unchanged. Collector answer_comparison_version normalized-decimal-exact/2 guards result/partial reuse. Exact/JSON/TreeDST/TATQA-specific policies unchanged.

Luna first audited older1278snapshot but mislabeled it recipe-v16. Root checked declared snapshot path537e372.../2306rows, requested exact ordered input assertion and rerun. Corrected scan10691savedrows includes792ref-v1/1400ref-composed/2306selectedgenerated and four static bundles6193cases;322accepted normalized checks,0changed positives,0QAspan checks. Old report/evidence preserved/superseded with hash. No migration hold needed from this scan; not all historical corpora audited.

Full-input audit of41 nextpool results led to six root-reviewed MuSiQue holds0/154/159/165/23/158: unspecifieddate/count, wrongHouse vs Panama term, two-leaderquestion with singlegold, windproduction vs potentialmetric, parent-ageclause scope, WWIIend label mapped to Mussoliniarrest. Original gold/evidence retained, no DPOnegative. Root withdrew proposed holds4/169 after full question chains: ScottishPanama1698 and LadyGagaFame createdby2017performer are supported. Static directoryquality-v13 excludes6cases,1267cases/3270approved/39cumulativeholds; allfour static32710approved, replacementchain explicit.

All530 eligible original cases attempted,489rawacceptedanyattempt incl reviewedv12/v17 (CommitPack287/TATQA183/TreeDST19); rawacceptance is not final admission. v17 fresh retries: threeMarkdown and datealias pass, expandedKhagan remains equivalence review. Solev18 boundary rollout100001; Luna100243/100068 migrated, Bonsai97277 pending. Next08:11fullcheck/inventorybatch; no tests/studenttraining.

## 2026-09-30 08:16 UTC — hourly health and consolidated recipe

Allv18workers deployed (Luna100243/100068,Bonsai100628). Actualfullcheck08:13:19UTC GPU99%,7457/8188MiB, container2.295GiB/6GiB, disk18.23GiBbeforeinventoryrefresh. Next09:13:19UTC. Nextpool106uniqueactualresults36rawaccepted/70reject (MuSiQue), not finaladmission. One Bonhardtimeout at1200s (broader-expansion346retained1,11savedreplies/23558completiontokens) is being independently audited for progress/repetition/sourcecontract before anybudgetchange. KnownLunafalseinactivity fixed/retried; no othernewtransportfailures. Fullincluded282MuSiQue sourceaudit delegatedread-only; rootreviewrequired beforeholds, oldlabels preserved.

Consolidatedrecipe-v18 currentpolicyincludes2362selectedgeneratedtrajectories/1931programs plus explicitlegacyreferenceinputs/allfourstaticbundles32710approved. No missingdefaults or includedqualityblockers. Finalstudentformatting/split/token/dedup/25%mix audits notyetcomplete; no trainingstarted/tests.

## 2026-09-30T08:37:15.270331+00:00 — exact computation and staged finalization

Exact-input computational/code outputs remain exact; no target injection or generic tolerance enabled. Contract19 permits omitted `value` on successful `return_result` only for an existing staged result, through normal completeness/type/transaction guards. Explicit values remain validated. Truncation feedback points to this shorter finish path when a complete result exists. Native TreeDST and file reference audits pass; missing stage, incomplete stage and wrong explicit values reject. Build passed, no tests/providers used in audits. Deployment pending reviewed journal-boundary rollout.

Six source-specific MuSiQue holds published quality-v14, preserved originals/golds and excluded from preference negatives. Root adjudications in hourly/musique-included-source-sweep/root-{first,second}-batch.json. Recipe-v18 explicitly stale pending refresh.

### 2026-09-30 08:48:57 UTC — v19 rollout preparation correction

First prepared queues/config preserved as superseded evidence. Root review caught mixed completed-group splitting (would rerun completed work) and retry budget/order changes. Prep-v2 preserves finished/current entries verbatim and places independent TreeDST retry first with original budgets. Frozen605files and queue hashes independently verified. Sole watcher110320 launched exact-PID journal boundaries; no extra teacher slots/server changes.

## 2026-09-30T08:55:07.855679+00:00 — reviewed invalid MuSiQue aliases

Future shared source builder removes exactly two source aliases in evidence-pinned v2 variants: Islam (primary Islamic mathematics) and Gun laws in Iowa (primary21 or older.). Pins original IR ID, full prompt, snapshot, primary, exact oracle shape and support-file hashes. Drift fails closed; idempotence validates full review metadata. Originals preserved, no blanket legacy-positive hold: exact recipe-v18 scan found three matching accepted rows, all valid primaries, no alias exploitation. Native variants accept primaries/reject invalid aliases; no model calls/tests/training rows. Evidence hourly/musique-alias-removal-v2. Frozen v19 unaffected; no immediate teacher rollout needed for future-builder-only adapter.

## 2026-09-30T09:12:18.619166+00:00 — second root-reviewed multihop hold batch

Eleven source-specific holds centrally published and directory staticquality-v15 filters1250cases/3253approved decisions,56cumulativeholds. Entire originalsource/gold preserved. Scopes include municipality versuscountry denominator, distinct Congo countries, same-name counties/villages, song versusvideo rumors, corporateoffice versusoperationscenter, diocese versuschurch body, unrelated educationinstitution, television series versusfranchise/soundtrack and salt-law Belgium premise versusIndia evidence. Sourceauditflags alone nevertriggerautomatic holds. Rootretained Nigeria2012 adult-HIVrate as natural suppliedmeasure despite broadcountry wording, avoiding blanket literalist exclusions. No DPO negatives; recipe18 explicitlystale until refresh. Teachersv19 frozenunchanged, futurequeueauditpending.

### 2026-09-30T09:22:07.154913+00:00 — hourly current inventory

Read-only health report joins exact304nextIRdigests and canonicaljournals;246uniquecompleted origins,102rawaccepted/144rejected,98current-native candidates. Zero new noncomplete finishes since08:13:19; zero futureheldroots (66heldqueueoccurrences allfinished). No urgentruntime rollout for11newholds. Recipe-v19 refreshedonce thishour,2473selectedtrajectories/2020programs, allfourstatic32693approved/6176cases, priorlegacyrefs explicitlyretained. Finalstudentadmission remainspending, no training. Next10:17:18UTC.

## 2026-09-30T09:43:36.988517+00:00 — complete source coverage, narrow holds and compressed snapshots

Full 282-case MuSiQue source audit finished; 84 review flags require root adjudication. Root requested a second question-consistency pass after spotting county-versus-county-seat and ocean-versus-atoll conflations. Natural band “born” wording and routine background links do not automatically justify holds. Four independently source-read contradictions/relation mismatches are held centrally and in static quality-v16 (1246 cases/3249 approved decisions; 60 cumulative holds). Golds/originals preserved; no DPO negatives. Recipe19 stale.

Future generated snapshots use gzip with immutable compressed-byte SHA and uncompressed content SHA (manifest/2); old plain artifacts remain readable and unchanged. Shared streaming reader covers admission, materializer, handoff/DPO and static inputs. Writer fsyncs gzip, ledger and manifest; publishes manifest last atomically. Actual saved native row yielded identical admission and materialized decision; malformed/truncated gzip rejected. Syntax checked, no tests/full snapshot or provider calls. This prevents repeated approximately2.5GiB plain snapshot growth.

Both Luna queues completed normally; old authorities are recorded as completed, Bonsai stays active. Preparing two source-qualified article-title filename pilot variants in separate queues for existing Luna slots. No extra worker, no existing queue mutation, no frozen runtime mutation.

## 2026-09-30T09:46:46.851176+00:00 — isolated article-title pilots

Root approved standalone title-path-v2-r2 after pinned-hash/idempotence/current-source checks and native controls. First physical H1 only; no arbitrary code/gold rewriting; unsupported numeric-path reference code fails closed. Two source-qualified variants launched into the existing Luna slots (133004/133005), unchanged budgets/caps/backoff/frozenv19, separate immutable queues/output/jobs. Prior exhausted queues retained. No general speed/reliability claim from two pilots; shared builder unchanged.

## 2026-09-30T10:06:13.215Z — source holds, measured compressed snapshot

Root confirmed11 more MuSiQue event/entity/ambiguity holds and SciFact694 malformed claim. Directory static quality-v18 contains1234cases/3237approved,72cumulativeholds. Root withheld typo-only SciFact admission: abstract does not resolve unqualified stronger/causal induce wording. A new source-derived narrower claim requires a new run, not migration of prior trajectory. Calibrations withdrew invented Watch Dem Roll video predicate and retained Bernini composite-authorship caveat; ambiguous biographical country links remain review, no unsupported false-source claims.

Recipe20 carries2542generatedteachertrajectories/2075programs, both792+1400legacysets and all4staticbundles; no missingdefaults/included blockers, finalstudent audits pending. Fullgzip hash verified67344798bytes vs523126712uncompressedbytes; earlier~2.5GiB estimate described observed diskdelta, not measuredsnapshotbytes. Originals retained; no training.

Found old supervisor complete exit0 versus15turn unfinishedpartial/emptyexport/no manifest. Futurecompletion accounting fix underreview; original retained, not answerfailure/DPOnegative. Two pilots completed, no live Luna process until nextreviewedqueue launch; Bonsai continues.

## 2026-09-30T10:11:47.549Z — reviewed minimum-age equivalence published

Exact source/prompt/full-base-digest/evidence pinned Iowa adapter adds21,21yearsold,21years for an explicit minimum-age question. Numeric age unchanged;20/22/range directions/removed title alias reject. Future builder applies after source groups and reviewed alias removal, before any optional title remap. Full-source/ref/group drift fails closed. Root independently verified unchanged actualtrajectory, actionledger, returnvalue21yearsold andfiles, native admission with3decisions. Separate approved migration artifact published; old result retained and original IR/request hashes pinned. Earlierreview-only .result artifact is superseded by this newer approved copy (same trajectory identity); futureprototypes must use review-candidate filenames to avoid automatic discovery beforeapproval. Recipe20 predates approvedmigration; refresh after hourly check. No provider replay or target injection.

## 2026-09-30 10:17 UTC — completion accounting and malformed question

Root approved supervisor output-accounting/2: exact frozen-native IR digests, matching saved terminal rows and merged export, batch manifest/hash/source/count checks required for complete. Explicit frozen-policy holds can skip on normal exit only; hard timeouts/inactivity remain failures. Only incomplete partials trigger log scanning. Saved accepted/rejected terminal rows pass; missing15turn partial fails. Syntax/native saved-artifact audits run, no tests. Running Bonsai process unchanged; next new supervisors use fix.

Root found original missing MuSiQue question asks which foreign group but gold380 is year. Original now source-held, rawpartial/history retained. Staticquality-v19:1233cases/3236approved,73cumulativeholds; allsourcecatalog replacements recorded. Preparing fresh distinct explicityear question, not original trajectory migration. Recipe20 stale after this1hold; refresh21 after hourly audit.

## 2026-09-30 10:52 UTC — exact numerical contracts, durable gates and failure visibility

Exact-input computational outputs remain exact; no target injection or generic numeric tolerance. Judgment-derived scores would require a separately declared bounded contract. Reviewed minimum-age aliases preserve the same exact age, rather than changing the target.

Root reviewed and committed bd7bffd: joint training requires current inventory-policy hashes and a version-2 mix report bound to exact ready corpus, token-audit manifest, renderer and audit code. Mandatory gate arguments follow caller overrides. Resume identity includes gate hashes; stale reports fail closed. Existing unbound mix analysis remains available. Reducer25% is a minimum gate, not a balancing sampler. No tests or training run.

Fresh clarified SciFact and explicit-year MuSiQue derivatives completed with SUPPORT/380 and seven admitted decisions total. Malformed originals remain held; no trajectory migration across changed prompts.

Read-only DPO carryflow audit found rejected teacher attempts absent from default pair discovery: positive snapshot selection excludes them, and ledger next-action text is advisory. Authorised narrow separate failure-candidate inventory implementation; retain causal/same-prompt/approved-positive pair gates, no automatic negatives. Delegation audit must inspect actual NL child invocation evidence, not only top-level delegate actions.

## 2026-09-30T11:16:36.494054+00:00 — reviewed title batch and failure carryforward

Root independently reviewed source chains; reduced24 proposed cases to13 beforelaunch. Hold reasons include ambiguous entities/answer sets, planned-versus-actual events, missing final residence and first-event date. Cleared overstrict analysis of explicitly requested Unicode subset, military branch OF a corps, and natural proportional answer. Corrected irrelevant Qing citation to exactYuan/Mongol/Jin excerpt. Launch186820/186821,6/7freshqueues, frozen19, existing limits/backoff, host output-accounting2. Priorproposals preserved, no extra workers.

Separate failure inventory committed e599ca7 with stream processing, exact original evidence, negative-ineligible labels and immutable manifest/carryforward receipt; no automatic pairing. Added --json opt-in to preserve old CLI consumers. New manifest type exposed catalogKeyError, fixed schema dispatch (b02e5b8); originalfailed/interrupted snapshot artifacts retained. Recipe21 published againstquality20 but subsequentholds require nextrefresh.

Verified actual childNL invocation contexts/ledgerIDs, not only explicitdelegate actions:2566approved childdecisions/48033auditedprojection(5.34%). Finalmix scope notcovered. Nine positive-selected trajectories failmaterialization correctly:7checkpointcaptures and2 missing invocation-attribution traces. No safe manualmigration found; excludedfinaltraining. No tests/studenttraining run.

## 2026-09-30T11:25:36.838Z — hourly check and stop unhelpful wording retries

BonsaiongoingGPU97%,138canonicalfinishes/0nonsuccesssince10:24,0futureunstartedheldcases. Luna13titlejobscompletedfullaccounting,1accepted/12rejected; source-read outcomes predominantlywording/specificity equivalence. Rootstoppedadditionaltitleonlyretrybatchpendingboundedsource-backedalias proposals. No fuzzy/substring/generalnumericvalue relaxation; wrong computionalresultsstaywrong. Actualrecordedanswers/originalgolds unchanged, no DPOnegativeassignment.

Sourcequality22nowcurrentexplicitpublicationfield;4static6150cases/32667approveddecisions. Nineadditionalholds sincequality19 and2unresolvedrelationreviews logged. Recipe22publishedwithdurablefailureinventory; finalcarryforward/hashreadinessauditongoing. CurrentbuilderpreservesoriginalCLIstdout, distinguishesmanifesttypes, carrieslegacy4static/allgeneratedeligibleinputs. No training.

Hourlyreader initialreportcaughtasyncgeneratorfor/ofreadfailure; rootfixedforawaitanddescendingauditsort, secondreporthasnoreaderrors. Bothpreserved; initialnotclaimzerooutcomes. Readerlateraddscompleteness/errorgate. DirectqueueoutputcountsdoNOTcoverallBonsai per-batchsavedjobs; supplementarycanonicaljoinneeded. Updatedcheckflatcounts/currentpolicy and next12:25:36.838UTC.

## 2026-09-30 11:47 UTC — deduplicate journal aliases in progress counts

Spool audit reconciles123Bonsaifinish events as60parentcase finishes+63member aliases; all60exactnative source/jobmatches exist,55admitted/materialized/237decisions,5rejects. Add13title and2derivatives=75distinctmodelresults inhour, not138newcases. Earlier138figure describesjournalfinish events only. Directemptyaggregateoutput isaworkingexport, notmissingdurablejobs. Futurehealth reports distinguishmetadataerrors, limitedoutputscope and canonicaljobresultcoverage.

Recipe22inventory/hash/carryforwardaudit16checks passes; root validatedreadyrecord currentpolicy. Stillnotfinalstudentrender/split/token/dedup/mix readiness. Ten scopedoracle-only equivalencereviews underway, actualteacherhistory/runtime/modelvisibleinputs unchanged; broad history-replay admissiongate stays closed.

## 2026-09-30T11:54:32.489Z — ten source-backed oracle-only approvals

Rootmatched10candidatepayloads against exactoriginalprovider/runtime/IRvisibleinputs, helperderivedaliasIR, unchangedoutputs/files/actionledgers/fulltrajectories/providerhashes. checkOracle+admitRow+materializerpassed10rows37decisions. Publishedapproved.resultfileswithhashboundrootapproval/provenance, originalsretained. No genericmigrationgatechange: these areunchangedhistories/runtimewithhiddenoraclecorrection, notofflinehistoryreplay. Futurebuilderintegrationpending. Recipe22staleonlybythesetennewpositives; nextfullhourlysnapshot includesautomatically.

Resolvedfalse traceintegrityalarm: recordedtrace_sha256hashescollectorcanonicalparsed events, rawJSONLfilehashis differentserializationidentity. Bothrecordedexplicitly, canonicalmatches10/10. Noactualtracechanges.

Rootcalibrated5Bonsairejections: metricselection(2percentvs0.7million) andaggregationunit(8meanannualtotalsvs4meanfourcells) arenotnecessarilyarithmeticdefectsunderambiguousquestions. Signeddifference±7 alsooperationconventionreview. Keepgold/originals; source-operationclarifications pending, noautomaticpreference negatives.

## 2026-09-30 12:14:39 UTC — reviewed source-builder contracts

Approved deterministic future-builder integration for ten bounded source-pinned wording equivalences and the exact Oklahoma three-event variant. Actual isolated buildTaskSources replay and native reference/admission/materialization audit passed; Node build succeeded. Original source holds and golds persist; no fuzzy comparison, numerical tolerance, target injection, history-gate exception, frozen-runtime mutation or teacher answer rewrite. Preserve original review-origin metadata despite its historical “not wired” field; current approval is recorded here and in HANDOVER. Static Oklahoma carryforward is prepared separately before publication.

Correction: the five rejected Bonsai rows contain three ambiguous operations, one missing unit, and one faithful text-span variation, not two proven arithmetic errors. Computations from explicit inputs/operations stay exact. Clarify source task operations with new IDs; retain originals and avoid negative labels for ambiguity.

## 2026-09-30 12:23 UTC — exact operation clarification

Applied central pending holds for TatQA456f8649-7ed6-41b7-a851-5c6487c0ba54, a018d9b5-a8ed-4b3d-b4cb-7fdf8692b17c and ceaf32b0-cfb8-4466-873a-c6fc51267a48. Full visible source supports alternative operation readings; preserve golds and raw attempts, no DPO negatives. Prepared explicit-operation prompts retain original values, units, sources and split groups. Root caught inherited external_source.source_id still pointing to held original despite derived source_ids; revised proposals must use consistent derived identity plus separate parent lineage and replay against current holds. Native reference counts are11/7/5, total23; no evidence was missing merely because earlier summary transposed counts.

## 2026-09-30 12:36 UTC — exact computed values and JSON consistency

Added opt-in `tatqa-answer-record-exact` for oracle normalization, output-file comparison and returned/written consistency. Exact decimal forms allow .70=.7 and reject .704=.7; units, sign, schema and duplicate keys remain checked. Existing explicit2dp task contract unchanged pending accepted-output audit. No target injection, numeric judgment bounds or changed answer values. Node build/native oracle+file+consistency controls passed.

Fixed JSON text-record consistency: returned whole record must equal answer.json. Existing explicit per-file alternatives now work under the strict JSON record comparator; read-only source files still exact. File comparison version bumped json-content/1→2; answer comparison version unchanged because the exact mode is new and identified by IR. Selected222static textrecords checked, allmatch. Native definition proposal (one literal source clause) admits/materializes3linkeddecisions with drift/value/unit/mismatch controls; publication still requires fullpayload/provenance review. Historical collector file-version must remain1; record separate current-review version2 rather than falsely retag the original run. Frozen19 untouched.

## 2026-09-30 12:46 UTC — bounded carryforward and exact pilot

Published one source-pinned complete-record definition alternate after independent fullpayload comparison and native checks; actual model answer/files/history and historical file-version remain unchanged. Three fresh clarified arithmetic questions ran with two Luna workers and exactdecimal mode: all accepted/admitted,12linkedapproveddecisions. Old ambiguous prompts remain held; no negatives assigned or targets injected. Frozen20 verified618files; original19 manifest unchanged. Bonsai exhausted priorqueue normally, nextbulkcampaign is explicit cross-teacher MuSiQue/SciFact coverage, not newunique sourcecases. Publishedquality23with3original removals+1Oklahoma exactvariant; allfourstatics6148/32645. Recipe23refresh audit pending.

## 2026-09-30T12:51:30.276969Z — resume Bonsai with new explicit coverage

Started Bonsai223773/frozen20/accounting2 with314sourcequalified model-diversity cases,212MuSiQue+102SciFact,4requests and4case batches. Preserve original330queue/journal finish evidence; no exhaustedqueue restart. Match existingBonsai byexactIRdigest ORsourceID acrosssavedresultinventory andsnapshot;79existing cases excluded,2rootunresolved sources excluded. Prior Luna data remainseligibleforadditionalteacherstrategies, notnewcasecount. Rootchangedprepared16casechunks to4casebatches, prioritized10reviewedvariants, correcting overlybroadpublishedIRbyteidentity claim. Queue/IRhashes andcurrent-policy/sourceconversion pass recorded inrootqueueapproval. AdditionaloldtimeoutTreeDSTalreadyacceptedelsewhere, no duplicate retry; CommaQA separatependingdiversity target.

## 2026-10-01 05:28 UTC — outage recovery and source-safe replenishment

Local teacher availability is now a prerequisite before starting a queue attempt; unavailable health journals service_wait without consuming a key or case budget. Failure backoff applies to both local Bonsai and remote Luna, retaining existing capped exponential jitter and provider retry deadlines. Semantic rejected results remain completed results. This prevents an unavailable local server from exhausting the queue. Original outage cause remains unknown; do not infer OOM. Preserve169 saved outcomes and recover145 unsaved cases only under new pinned queue/output paths. Four Bonsai requests retained.

Published source-reviewed staticquality24: remove unsupported MuSiQue source82 and ambiguous signed-original374c, retain1220 original records and add one explicit signed derivative. All history/old manifests retained; no source-ambiguity negatives or fabricated DPO pairs. Catalog/recipe carryforward refresh follows publication. Two Luna workers launched on256 independently reviewed workflow-only fresh sources; both identity fields scanned, all prior attempts excluded. These are additional teacher strategies, not missing static case coverage. No tests/student training; frozen20 remains immutable.

## 2026-10-01 07:49 UTC — persistent checks and utilization safeguards

Added singleton hourly health monitor with60second availability heartbeat, exact batch spool reconciliation and authority-baseline race check. It reports idle capacity for agent review; it does not autonomously weaken admission, launch unreviewed work or perform AI repairs. Full check now68recoveryrows55admitted13rejected, no transport failures; Luna250/256accepted. Nextsource-safe queues prioritized to keep both teachers useful.

Missing-server watcher previously defaulted to6slots and could forcibly replace a live loading container after failed health. Changed default to4slots/53248context, singleton flock, bounded exponential delay and restore-only-if-container-absent; existing container health failure is reported without destructive replacement. Shell syntax parsed, no tests. Stop watcher and hourly monitor during intentional shutdown. Catalogquality24 reviewed: allpriorlineage plusfive newedges preserved. Recipe24 explicitstale/missing-input blockers, freshsnapshot refresh pending.

## 2026-10-01 08:12 UTC — reviewed successor handoff and streaming identities

Launched1024freshLunacasesunder2workersafter independentfullhistory/sourceIR/gate/currentnative audit; earlieremptyproposal/report-pointerdiagnosticsretained. StreamlargeJSONL andnativeauditboundedchunks; admitRowrequiresfullsavedresult, notIRwrapper. Sixpriorworkflowrejectsaremodelanswererrors,no oraclechange.

Staged256caseBonsaisuccessor(192reducers)behind currentrecovery. Controllerclaimslaunchonce, requirescompletepredecessoraccounting andrehashesallsource/queue/runtime/policy artifacts; no rerunsor unreviewedlaunches. Addedstreamedrecord-digestsCLI toexposeofficialcollectorcanonicaldigest andkeeprawline/fileSHAidentitiesseparate. OriginalBonsaiproposalmetadatafixrecordedinmanifest-v2 only; IR/queueunchanged. ClearedunusedElectronarchivecacheonly(1.416GiB), alltrainingdata/modelweights/historyretained.

Inventoryfoundlegacyfalsepositivezero-fallbackturns(12)innewdefaultfolder-revisionslane. Keepreadinessblockedwhileexactproducer-boundexclusionandpipelineguardareprepared; schema-v2aloneisnotqualityadmission. No raw historyrewrite/fabricatednegativepairs.

## 2026-10-01 08:43 UTC — Keep both teachers supplied

- Staged independently reviewed successors: Bonsai256 and Luna541. Luna continuation is per-slot so a completed worker does not wait for the other512-case queue. Preserve the two-request Luna limit and four-request Bonsai configuration.
- Shared authority lock coordinates hourly-monitor and successor updates, avoiding lost worker IDs during simultaneous handoffs. Controllers389121/389122 and monitor389123 supersede earlier PIDs; stop all during intentional shutdown. Syntax parsed, no tests run.
- Full legacy source review expanded the folder-revision hold from12zero-fallback turns to all23turns, and found10additional v2aggregation turns whose validation subset happened to pass. Candidate review must preserve source-specific retained/excluded partitions and cannot replace old history with successful fresh v4 history. Originals remain immutable; no fabricated DPO labels.

- 08:45 disk safety: reclaimed1.451GB inactive downloadable Homebrew/pnpm caches and generated thumbnails older24h, no open handles. Audited each deletion; training/model/original artifacts preserved. Approximately2.8GB free afterward. Hourly agent investigating sudden consumption.
