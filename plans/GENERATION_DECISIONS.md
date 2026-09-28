# Generation decisions — 2026-09-27 continuation

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
