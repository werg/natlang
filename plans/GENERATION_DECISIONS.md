# Target accounting and successive batch approval — 2026-10-03 14:25 UTC

- A single global unfinished-source-task count cannot be derived from the artifact inventory. Report separately: current provider attempts, deduplicated derived-task candidates, distinct original source coverage, held sources, and attempted repair candidates. The v9-v3 universe leaves3391 alternative compositions after completion; local assigned attempts were255 at14:20. These reuse sources and must not be summed as distinct source tasks. A separate read-only target reporter and persistent registry proposal are in preparation.
- Extend the approval helper's explicitly allowed campaign/predecessor map with v9-v4, requiring both v9-v2 and v9-v3 IR and import evidence for collision checks. Preserve all exact runtime/native/input/source/hash/root-approval gates and immutable old approvals. The new batch remains unapproved until its full packet passes. Python syntax was parsed; no tests or provider calls for this helper change.

---

# Source-copy filter repair — 2026-10-03 13:20 UTC

- Blanket `*.lock` exclusion also excluded source dependency lockfiles. Add a narrow allowlist for data-tree `yarn.lock` and the tracked public deno-std dotenv fixture. Copy and verify the 124 existing exceptions without interrupting the active broad copy; reload the recurring service only after that pass finishes. No credential contents printed and no tests run. Preserve the fixture upstream revision and local/remote hashes in the repair receipt.

---

# Pipeline replication and importer decisions — 2026-10-03 12:25 UTC

- Replicate the entire retained pipeline into the DGX development checkout, including original sources, intermediates, failures, review and lineage evidence, recipes and runtime dependencies. Historical provenance is retained unless explicitly reviewed as obsolete. Copying a rejection does not admit it for training. Source changes use Git; unreviewed development work is separately preserved with hashes.
- Importer 47e1560 waits for successful final transfer/import and exact finished-assignment coverage before exit. Record live source-vanished races separately, exclude temporary write files, and use an rsync I/O idle limit so progressing large copies can finish. Completed v9-v2 exact coverage and automatic v9-v3 handoff were observed after deployment.
- Current DGX successor config is immutable v4 (b826a979d335a68d54e41b9502238f2a0f313a71738f404527569e0b04c3ff39), watcher-v3. Both Luna v44 successor slots now run under the owned-child lifecycle fix. Retain superseded configurations and failed lifecycle evidence; never arm another duplicate.

---

# Ling training efficiency decisions — 2026-10-02

- Latest user explicitly prioritizes GPU utilization on both machines: keep DGX Qwen generation busy, local RTX4060 on real350M training; CPU data preparation in parallel. Start an audited-data continuation while broader data is being rendered, then hand off at a recovery-checkpoint boundary. Preserve first-pilot/evaluation identities; defer optional Ling GPU benchmarks while useful training runs. Correct pinned model cache is `models/hf`; a wrong mount caused a pre-load failure, not missing weights requiring redownload.
- Ling download finished authenticated,32weightfilesLFSverified. Prepare a separate exact-source-patched model on externalDGXcache with linked weights; no GPU load and no extra15GBweight copy. Older local `ling3-tiny-grouped` already has grouped/Liger optimization; audit/reuse that earlier work before inventing another production dispatcher.
- **32c87f4:** enable source/configuration-hash-bound short-training and completion-logit fixes through a separate prepared model directory; preserve original code/weights and link weights to avoid duplicating 15.8 GB. Actual reduced BF16 CUDA KDA+MLA+MoE + PEFT tests match full-projection loss/gradients exactly and short five-token backward passes. Integer projection only; reject unreviewed MTP/tensor projection requests. No padding or sorted routing changes promoted. Full Ling/Spark throughput remains to measure.
- Non-root KDA containers need an explicitly writable persistent `TRITON_CACHE_DIR`; the inherited workspace cache caused a permission error. Scoped cache setting fixed the probe; no broad ownership changes or weight edits.
- Corrected right-padding numerical test still exceeds the original max-logit bound. Keep it held and investigate kernel/layout numerics. Reduced sorted GPU/custom-LoRA speedup 1.46× is a prototype measurement, not full PEFT/model evidence.
- v13-r2 preparation produces 105,869 train /5,432 heldout decisions (~4.88%), addressing v12's inadequate80-row heldout set. Broader LFM experiment uses reviewed model-neutral data with source-component isolation and explicit model-specific rendering/audit. First100-step model's lower loss did not establish task success; whole-program0/4 and turn-action1/12 remain honest baselines.
- **db70db2:** `--expert-rank` now also applies to individual routed expert linears. Previously the stacked-only regex silently ignored Ling's `experts.N.projection` modules. Shared experts retain the ordinary rank; no automatic reduction in the default rank. Actual PEFT tests check rank/alpha rather than regex alone.
- Custom models without per-layer checkpoint flags reject `--retain-every-n-layers > 0`; native full checkpointing remains available. This replaces a silent no-op with a clear error. Training gradient clipping reuses the existing trainable parameter list.
- Ling's pinned custom forward ignores `logits_to_keep`; full-label smoke tests are insufficient for our shortened completion-label training path. Require actual trainer loss/gradient equivalence before enabling it. Reusable fixes must be bound to the reviewed upstream source hash and preserve the original snapshot.
- Sorted MoE dispatch is still experimental. Preserve router precision, expert/LoRA coverage, weighted routing order, and zero-gradient optimizer semantics for empty experts. Two-step reduced CPU AdamW equivalence passes after adding empty-view zero dependencies; GPU/PEFT throughput proof remains required. Do not copy all frozen expert weights each forward or reuse the no-grad inference dispatcher as training code.
- The KDA padding CUDA comparison currently exceeds its provisional valid-logit tolerance. Do not label it passing or loosen the tolerance to fit the result.
- Download the pinned BF16 Ling snapshot into DGX external cache with the HF token, no GPU allocation. Do not interrupt current Qwen teacher generation for the download or claim its inference speed establishes training throughput.

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

# Storage and provider continuity — 2026-10-02 18:53 UTC

- User authorized storage cleanup. Reclaimed unused aged Docker build cache and dangling images rather than move bound data; actualavailable153.48GiB (+142.85GiB), current training image retained. No data/model/volume/container deletion. See storage receipt for actual filesystem figures, not Docker shared-layer accounting.
- A run-local OpenRouter supervisor feature had never reached canonical code, causing successor CLI rejection. Ported provider request-config handling to canonical runner with explicit provider/JSON validation and forwardingtests; commite728e95,16 queue tests. Preserve no-call setup failure, retry same unattempted queue with known compatible helper; futureplanspin canonicalnewversion.
- Existing-data SFT recipe must hash source/hold/DPO metadata for provenance without interpreting it as admitted training content. Maintain explicit roles, carry-forward resolutions and standard required-default/quality checks; no blanket inventory override.
- DPO61 distinct causal pairs stay visibly pending student rendering/dedup/sourcegroup split/token audit. Raw failure inventories do not automatically become DPO negatives.

---

# Continuity repairs — 2026-10-02 18:32 UTC

- Resume real Luna/Bunny work after finite queues, with current proof and history suppression. Bunny256 now runs; Luna20 runs on private fixed x64v38. Preserve setup attempts and distinguish no-model infrastructure failures from teacher negatives.
- Create collector output/log/jobs directories in the supervisor before attempt start; fresh queues must not rely on manual directory setup. Commit2343bb6,14 tests.
- Root independently verified private x64 v38 closure and Node24 pin; don't interrupt healthy Luna2 work to deploy a fix without a genuine drain mechanism. Use fixed runtime for subsequent reviewed work.
- The npm corruption was caused by this task's hardlinked build clone, not a model/provider failure. Local v38 was atomically restored from exact DGX bytes and fully audited; v37 remains quarantined. Physical clone/inode-isolation helper committedcca6938; seal verifier now refuses multi-link files. Preserve old receipts and correct misleading claims of isolation.
- Restore complete standalone host dependency lock without package.json changes or resolved upgrades. Preserve cross-platform optional entries, including DGX ARM esbuild, rather than publish a Linux x64-only lock. Commit46fb511; root workspace lock optional coverage separately under review.

---

# Course changes — 2026-10-02 18:15 UTC

- User correction controls: only Bonsai stops; Luna and Space Bunny must continue. Queue preparation alone is not worker continuity; report actual processes and boundary transitions.
- Refuse launching from a modified sealed dependency tree. Hidden npm lock mutation blocked the new Bunny queue; preserve evidence, restore only exact independently verified bytes and permissions, re-audit full closure. Do not silently alter frozen manifests to admit corruption.
- Separate all-corpus current policy runtime from campaign-frozen v38 (five of seven policy modules differ). Generate coherent current dependency lock in private clone, compile current source there; never mix frozen physical dependencies with a different lock or mutate live main dependencies.
- Fix model serving termination as well as corpus rendering: derive the assistant end token from the closed chat template, require one reversible token ID, explicitly pass generation EOS, retain tool markers and count the final terminator in usage. Commit35cbf4f;10 focused tests pass.
- Published Qasper/static v26 is scripted annotation-directed reference construction, not independent model reasoning. Preserve source/gold identities and prior raw records; unsupported direct-answer decisions remain held.
- Before history suppression, map old case IDs through approved current transformations and verify current admission/materialization. Thirty previous short repair cases already had valid Luna positives; unknown/ambiguous transformations stay blocked rather than guessed.

---

# Generation decisions — 2026-09-27 continuation

## 2026-10-02 campaign runtime versus full-corpus policy

- The fixed v38-r2 runtime is a derivative of the sealed ARM37 campaign runtime; it is approved for its reviewed queues, but five admission/source-conversion/collector modules differ from current canonical training policy. Do not label campaign-policy checks as full current-repository policy checks. A full-corpus training runtime must be compiled from current canonical source in an isolated directory; leave live/frozen trees unchanged.
- Root initially suggested reusing v38 for the data-only DAG, then retracted this after comparing module hashes. Retain the course change: current static v26 policy pins and source holds must be checked in the training runtime. The DGX951 successor also needs a separate current-source-policy compatibility sidecar before gate rearm.
- All32 recent Luna repair sources already have current canonical task variants. Old:v1 historical repair selection bypassed existing TATQA numeric/evidence-scale instructions; preserve attempts, use a guarded current-case preference/lineage check before future assignment. No silent trajectory or gold migration.
- Workflow availability census pins4803 current workflow tasks, admitted snapshot plus730 recursive recent results and1024/951/306 active or pending assignments. After source-level active overlap and same-teacher successes,531 Bunny and20 Luna candidate IDs remain before final historical/native/source review. Receipt runs/teacher-supply-availability-20261002/workflow-availability.json; availability is not generation authorization.

## 2026-10-02 delegate lock cleanup and continuation

- Confirmed infrastructure bug, not an answer-quality rejection: malformed delegate returns type throws during child setup while retaining a previously acquired folder transaction. A later same-path delegate waits; Node can exit0 without a terminal result. Preserve old Space Bunny index211 partial; no SFT positive or DPO negative. Audit: runs/delegate-deadlock-audit-20261002/audit.md.
- Fix all supplied/acquired child invocation leases on preflight/setup errors; clarify returns type expressions in tool schema. Frozen runtimes stay immutable; deploy a new reviewed version to future queues.
- Luna1 repair continuation32 is active; Luna2 continues. Bunny33 credential preflight fixed via owner-only existing OpenRouter env file, launch waits fixed runtime. Index211 is already assigned to current DGX1024 and Luna2306, so do not add a duplicate fresh attempt.
- Static v26 visible-Qasper candidate independently hash/partition/native reviewed and publication authorized; source-held13 remain explicit, unsupported Qasper final returns300 remain unapproved. Full training data build remains resource-limited; actual local smoke and222tests passed.


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

## 2026-10-01 12:07 UTC — Restore response to hourly alerts

Mechanical hourly audits persisted, but root follow-through stopped when its turn ended. Luna storage-paused queues are now resumed with original journals; terminal successes/failures preserved, only unattempted entries run. New rolling plan binds resumed PIDs and precisely hashes four reviewed empty failed finish records so those excluded failures do not block unrelated approved next work. Unreviewed incomplete work still blocks. Preparing approved automatic storage-pause recovery to avoid dependence on root for this specific reversible event. Next full sweep13:06:58UTC. No tests/frozenruntimechange.

## 2026-10-01 12:22 UTC — Prevent repeat storage idle and paging waste

- Explicitqueue-scopedautomaticstorageresumeenabledforfivealreadyreviewedqueues, after4GiBfreeandfullartifact/authority/duplicatechecks. Monitor490927. Rollingcontroller489449bindsqueueidentityacrossPIDchanges; fourreviewedfailedattemptsremainexcluded. Intentionalshutdownmuststopmonitor/controllers.
- Source-boundlegacyguardrootapproved33heldturns/59retained; oldproducersandcandidateevidenceimmutable. Resolverapprovedstatusbugfixed; freshrecipe/readinesspending.
- Futurepromptguidance covers actualslicedelegation andminimalleafedits. Currentfrozen20unchanged. ExistingboundedMarkdownEOFvariantmustbeappliedtonextfreshsourceIRs beforegeneration; legacyqueuedIRs otherwisekeepcreatingavoidablefalse-negatives. Historicalrow74keptunchanged/heldpendingnewteacherorexplicitoracle-onlymigrationreview.

## 2026-10-01 12:35 UTC — Distinguish infrastructure, source and model errors

- MissingBonsaimember48collectorloghasexplicitENOSPC; repaironlyunsavedmember, preserve49–51.
- Narrowed11MuSiQueholdproposalsto9candidates: withdrawtennishold(actualpassagesupportsDjokovic), keepGoshenreviewonlyuntilalternativecompletechainisdemonstrated. No policypublication/DPOlabels.
- Minimalfutureimproververifierguardsrequiredeclaredtrain+validationquality1/gates. Syntaxparsed, no tests/replay; twoown editsleftinuntrackedunrelatedfile, patchreceiptpreserved. Freshdraftrecipe26refreshunderway.

## 2026-10-01 12:48 UTC — Apply guidance at reviewed campaign boundaries

Preparedruntime21as20clonewithonechangedpromptmodule/threebullets; all618parentfilesverifiedandallotherexecutionsemanticsunchanged. Next541LunaIRsindependentlymaterialize5862decisionson21; rollingplanv5/controller500258activates21onlyaftereachreviewedcurrentqueuefinishes. Currentworkersremain20. Futurestorageapprovalsrebound21; originalsimmutable. NextBonsaipreparation21+modernscopedMarkdowncontracts.

Currentv4improvementlaneevolvedto76rowswith10migratedfalse-positiveaggregationturns. Rootheldwholelaneuntil66-rowcandidatepartitionreview. Versionmigrationdoesnotestablishsemanticcorrectness; earliergoodv4empty-reductionhistorycannotapproveotherproducers. Recipe26draftblockeduntiladjudication/carryforwardpublication.

## 2026-10-01 13:07 UTC — Validate follow-through and automatic carry-forward

- Hourly monitor completed another real sweep: 28 new Bonsai results, 26 admitted, two rejected; both Luna slots progressing. Root remains active to investigate alerts. Next14:07:46 UTC.
- Approved exact66/10 partition of evolving v4 lane after all producer/case reviews. Across legacy reviews,43 physical turns excluded and289 retained. Recipe27 remains draft and uses explicit overrides; default pipeline automatic inclusion is being investigated separately.
- Root independent scan rejected Bonsai512 v3 proposal for a previously attempted source; rebuild exclusions rather than weakening freshness checks. Originals/proposals remain preserved.
- Generic successor now follows authoritative queue/journal across resumed PIDs, allows only reviewed exact failed events, and locks final liveness checks, claims, launch and authority reconciliation together. This prevents storage recovery racing a handoff. Syntax parsed; no tests run.

## 2026-10-01 13:12 UTC — Repair default inventory replacement accounting

Required default source paths now resolve through strictly approved catalog replacements only when manifest and physical row hashes validate and the replacement is present in actual recipe inputs. Default recipe28 carries all289 reviewed turns and resolves all nine required inputs without overrides. Root independently recomputed inventory and checked lane hashes/counts. Source-hold publication and final renderer readiness remain pending. No tests/training run.

## 2026-10-01 13:54 UTC — Publish exact holds and monitor handoff health

- Published nine independently confirmed MuSiQue source holds and exact static quality25 partition. Preserved sources, golds, raw trajectories and prior publications. No source-held failure becomes a DPO negative.
- Frozen22 changes only source-review source/module relative to21 (eleven literal records); active teachers20 continue unchanged. Re-admitted541pending cases,5862decisions; controller554491 and future storage approvals now22. Stopped only waiting controller500258 after checking successor record workers[], not active workers.
- Monitor565131 flags waiting-controller death, changed plan/current policy/helper pins, and unreviewed latest failed parent finishes each minute. It does not retry semantic failures. Actual reviewed-controller inspection passed; no tests run. Full hourly audit remains14:07:46UTC.
- Root verified default recipe29 carries6138static cases,3768model trajectories and289reviewed turns. All held positives absent;1626failure candidates retained/ineligible as DPO negatives. Final rendering/token/split/dedup/mix gates pending. Current-vs-frozen module differences are explicitly recorded; original execution histories remain bound to their original runtime.
- Complete provenance-based history scan shrank fresh traditional reducers to92. Preserve50%nextbatch reducer target by preparing164official typed workflow directory-v4 cases with native source conversion and256primitives, not reusing obsolete broad output schemas or pretending attempted cases are fresh. Modern adapter preparation remains unapproved until root audit.

## 2026-10-01 14:21 UTC — Approve fresh successor; preserve exact static bytes

- Root independently approved512freshBonsai targets with50%reducers and separately authorized exact same-IR repair48 after ENOSPC. Controller580812 waits for current256queue; generic handoff allows only the exact reviewed failed parent hash and pins runtime22/all618files plus immutable source artifacts/current admission. Mutable default pointers are not launch pins; underlying reviewed immutable inputs remain pinned.
- Added sixth scoped storage-pause recovery approval; no semantic retry or saved-success rerun. Monitor584649 now validates controller argv before launch-record creation as well. Both future controllers have zero actual health alerts.
- Rejected first workflowquality4 staging publication because unchanged4,639rows were reserialized, risking stored source-row hashes. Require exact retained bytes and source-bound one-for-one164replacement mapping; no additions or hidden gold changes. Preserve failed draft for audit.
- Strengthen reusable provenance inventory to hash exactly parsed bytes, check filesystem identity/timestamps and fail closed on stream/catalog errors. Existing pinned inventory remains immutable; new integrity scan is separate.
- Root reference audit memory fix streams all records but retains only requested targets/factory rows; no tests/models/training. Full14:08sweep zero new infrastructure failures; continuing hourly15:08.

## 2026-10-01 14:37 UTC — Publish typed static replacement and explicit archive scope

- Root published164exact source-bound typed workflow reducers in quality4; preserve4639othercases/27822otherturn byte rows. Source snapshots/IDs/groups/revisions/license/split/golds unchanged. Default pointer/catalognowincludeupgrade automatically;6138staticcases/32652decisions. Student readiness pending, no training.
- Correct prior concern: canonical parsed-object turn digests are insensitive to row formatting; reserialization was an exact byte-preservation failure, not a proved broken turn source_ref. Preserve audit correction.
- Hardlink approved immutable artifacts to avoid duplicate disk use. Losslessly archive only rejected unpublished draft largefiles with original/archive/roundtrip hash receipts; do not remove canonical source data or training positives.
- Root supplemental integrity scan confirms512fresh targets have zero historical/queued/originalpartial collisions. Scanner dependency-link exclusions explicit; unmatched directory/artifact links/readchanges failclosed. Prior approved inventory/queues/plans immutable.

## 2026-10-01 14:50 UTC — Explicit static replacement chains and default recipe30

- The publication entry/default pointer did not alone satisfy historical carry-forward inventory. Add exactly three workflowquality3→quality4 artifact mappings; preserve all old paths/history. No weakening strict retained-subset legacy turn approvals.
- Root independent default inventory/hash/gzip/sourcehold audit passes:6138staticcases/32652decisions,3887generatedpositives/3006programs,289reviewedturns,1630nonnegativefailurecandidates.9requiredresolved,0unreviewedomissions. Recipe30draft; finalstudentgatespending, no tests/models/training.
- Rejected unpublished draft losslesslyarchived with all three roundtripSHA matches;2.334GBsaved, canonicaldata untouched. Approved hardlinked artifacts0444 readonly; no future inplace edits.

## 2026-10-01 15:12 UTC — Investigate stall rather than hide it; correct inactive guidance

- ReviewLuna293asexactincompletefinishonly, notinfrastructureprooforsemanticnegative. Preserveraw17-responsepartial/emptyoutput. Same frozen20 diagnosticreplaymatches17requestSHAs/reaches18threquestin354ms withzero modelcalls; provider/planningstallsuspected,historicalphaseunknown.
- Newcontroller617447planv7 allows5precisefailedfinishhashes; activeworkersunchanged, no caseautomaticallyrerunoradmitted.
- Foundmissingproviderdeadline/cancellationforwarding. Concrete source-onlytimeout/phasepatchbeingreviewed; preserveoriginalfrozenruntimes and delayedtransportretrybehavior, exclude implicitcase retryonnewtypeddeadline. Future23handoffrequiresfullrootreview.
- Found prior3guidancebulletschangedunusedAPPROACHprompt; supervisorneverenabledoptionalguide. Correct claim andwireminimalguidanceintoactualdefaultprompt in future runtime ratherthantreatingunusedtextasdeployedfix.
- Hourlysweep15:08deltaBonsai36results/35accepted1reject/158decisions; one newLunastall investigated. Next16:08UTC, continuingactivefollow-through.

## 2026-10-01 15:33 UTC — Bound logical provider cycles and preserve fatal timeout evidence

- Attribute Luna293 cautiously: provider stall suspected, exact historical phase unknown. Missing cancellation/deadline handling is a confirmed system bug.
- Choose180s request plus250s shared prepare/plan/action cycle and15s cleanup, below300s inactivity watchdog. Do not alter supervisor or fake activity.
- Explicit typed deadline propagation through planning fallback and a collector fatal-error latch are required: native eval can turn child errors into tool observations. An incomplete deadline attempt must retain its partial and cannot become an admitted success or implicit case retry.
- Initial runtime23 staging is provisional until cycle/latch source freeze and independent review. Active teachers and frozen20/21/22 remain unchanged.

## 2026-10-01 15:42 UTC — Supersede proposed blanket deadline defaults

- User is skeptical of blanket timeouts and asked about language semantics. These controls belong only to resource management in teacher collection.
- Supersede15:33 proposed180s/250s defaults before deployment: deadline settings become opt-in, with explicit provenance only when configured. Preserve cancellation, phase logging and terminal handling of configured deadline failures.
- Existing300s supervisor inactivity limit cannot prove provider stalling. Partial observed intervals are not isolated request latency; do not use them to impose universal cutoffs or semantic rejection labels. Runtime23 staging remains provisional; active generation unchanged.

## 2026-10-01 15:50 UTC — Preserve colliding legacy lanes explicitly

- Two approved legacy replacement lanes reuse25IDs with different contexts. Rehearsal-ID dedup could silently drop examples; derived lane/source/review/row identities preserve both with original lineage.
- Root verifies125unique derived IDs,25shared original IDs, exact payload round-trips and unchanged source/review hashes. Wire only exact included approved lanes; preserve original artifacts.
- Two historical unlinked generated trajectories remain linkage-review pending and emit zero training turns; do not drop other attempts of the same program or fabricate ownership. No model/test/training run.

## 2026-10-01 15:58 UTC — Approve runtime23 with opt-in controls only

- Root verified621frozen hashes and1054scripted native references with0reject/unlinked. Future queues use actual minimal default guidance, phase telemetry and cancellation; deadlines disabled by default and in plans.
- Correct inherited22prompt source/dist mismatch without changing old snapshot:23 preserves existing compiled optional guidance and synchronizes its source. Preserve all unrelated dirty changes.
- Replace only waiting controllers with671283Luna/671284Bonsai, immutable plans and future23storage approvals. Current20teachers/active recovery approvals unchanged. Exact historical failure exclusions retained, no blind retry.
- Keep real stream-progress monitoring a separate proposal; no fake activity, universal latency cutoff, tests or model calls.

## 2026-10-01 16:11 UTC — Hourly follow-through and genuine progress candidate

- Actual16:09sweep0new non-success finishes; Bonsai16/16new exactresults admitted, main+111rows/+109admitted/+2Luna rejects under audit. Next17:09UTC, rootactive.
- Stage separate source-only aggregate Pi stream observation, never model content or fake progress. No watchdog refresh/defaultdeadline or23snapshot mutation. Rootreview/freeze pending.
- Keep recipe31draft until complete default carry-forward and source-policy audit; training remains off.

## 2026-10-01 16:20 UTC — Correct rejection scope and review fresh recipe

- Correct16:11 attribution: two main rejects include one Bonsai result from still-active batch184 and one Luna case343, not two Luna. Finished Bonsai spool16/16accepted has narrower scope. Wrapper/polarity errors repeat known classes; preserve gold/history, future typed schemas/minimal predicate guidance address ambiguity.
- Root independently recomputed recipe31default inventory:9requiredinputs,41oldpaths accounted(40replacement/1approvedhold),2laneidentitystages correctly consumed. Streamed all4037positives through currentadmitRow and ALLcurrent sourceReviewReason predicates:0held, compressed/content hashes exact;1632failurecandidates allnonnegative. Draftonly, finalstudentgatespending.
- Root verified runtime24:621files/exact10source/dist/typechanges; genuineaggregate streaming observation, unchanged prompts/watchdog, deadlines off. Native referenceproof/handoff review pending.

## 2026-10-01 16:23 UTC — Approve aggregate observation and preserve training scope

- Bind next queues to24 with genuine aggregate Pi stream observation, no content/fake heartbeat/watchdog extension/default deadlines. Preserve all current20teacher work and oldsnapshots/plans.
- Root independently verifies621files and1054native references; actual24providerstream behavior remains to inspect at first live handoff. Waitingcontrollers701232/701233 and future24recoverypins have0healthalerts.
- Publish recipe31 as inventory draft only after defaultcarryforward/4037positive fullpolicy/gzip/1632nonnegativefailure audit. Finalstudentgates remain pending; no training.
- Preserve two unlinked historical rows as explicit linkage holds; sourceevidence doesnot support deterministicownership. Other attempts remain available.

## 2026-10-01 16:47 UTC — Keep timeout evidence separate from correctness

- User skepticism of blanket cutoffs is explicit. No default provider timers or language deadline semantics. Existing300s supervisor watchdog remains a collection limitation to investigate, not proof of stuckness; no automatic timeout-negative labels.
- Preference audit isolated18causal pairs;3replay mismatches held. Do not claim full recipe preference stage executed or final DPO readiness.
- Reducer share measured by approved learner operations: static23.20%, combined pre-render30.25%; legacy program editing excluded. Final25% gate still pending.

## 2026-10-01 17:38 UTC — Remove blanket silence cutoff; observe actual provider output

- Default no-observation cutoff is disabled; optional `--no-observation-seconds` reports unknown stuckness. Breaking status change for enabled silence ceilings: `no_observation_limit`, replacing `inactivity_timeout`. Wall budgets remain collection resource controls. No deadline enters the language and no timeout alone creates a negative label.
- Count increasing saved replies, local decoding, and actual nonempty stream bytes; report pending phases and retry waits separately. No fake heartbeat or semantic progress claim. Signals are job-level, not per-request health.
- Refresh pinned waiting controllers and all six storage-recovery approvals under the authority lock. Keep current running teacher processes and all runtime snapshots unchanged; policy activates at their next approved launch/restart. Preserve exact legacy supervisor bytes for provenance.
- Source review and Python syntax checks only, no tests. Live runtime24 stream path remains unobserved until handoff. Mutable dependency closure is a separate confirmed reproducibility gap, not proof of the historical stall's cause.
- Correct the subset audit's copied recipe digest explicitly. Hold three model-visible preview mismatches despite unchanged console outputs; no casual equivalence or pair promotion.

## 2026-10-01 20:29 UTC — Correct hourly monitoring claim

- Background audit timestamps are not agent check-ins. Monitor writes reports only and cannot deliver this chat's investigation on its own; no wakeup was configured. Root ended its active turn, causing the requested human-facing hourly follow-through to stop.
- Record missing18:09–20:10agent review explicitly; generation stayed live. Require verified wakeup/delivery or an active turn before promising scheduled agent investigations.

## 2026-10-01 21:02 UTC — Space Bunny teacher and fresh rejection audit

- User supplied OpenRouter credentials in private `~/.config/natlang/openrouter.env` and explicitly asked us **not to send `enforce_distillable_text`**. The flag is absent from the actual provider override. The key is never in repository plans, command arguments, provenance or printed diagnostics. Authenticated key metadata check succeeded.
- Added one independent OpenRouter teacher worker, one concurrent request including children, on `stealth/space-bunny-alpha`. Isolated queue: `runs/space-bunny-20261001/queue-preparation/revision-2/openrouter.queue.jsonl`, 64 unchanged approved CommitPack IR cases (51 Markdown, 13 YAML/YML), first 8 interleaved. This is additional teacher coverage of existing cases, **not 64 new case identities**. All 64 independently replayed native golds under frozen v25, admitted with 256 linked reference decisions; no model calls during reference review and no gold/source edits.
- OpenRouter routing allows only the free Stealth endpoint, `max_price` prompt/completion zero, no fallback, supported parameters required. Public endpoint advertises retirement 2026-10-05: startup and each new case stop then. The old SDK catalog lacks this model, so inherit only compatible OpenRouter transport settings from `qwen/qwen3-14b`, override model metadata with Space Bunny's published values; the actual request model is always `stealth/space-bunny-alpha`, never Qwen.
- Capability adaptation: Space Bunny supports automatic tools but not forced/required tools or seed. Override wire `tool_choice` to `auto`, omit wire seed, retain native tools/oracles and collection seed in provenance. Disable separate forced planning calls, retain provider reasoning. Use low reasoning, max 8192 output tokens per request, context budget 16384, 40 turns, 384 model requests and 1200-second case collection budget; no silence/provider deadline. These limits are collection resource controls, not language semantics. Provider transport failures/rate limits receive delayed exponential retries and queue failure cooldown. Pause after three consecutive collection failures to prevent a bad endpoint consuming the queue.
- New `--provider-request-config` accepts JSON transport controls and records them in teacher provenance. Frozen v25 derives from v24: only owned provider-control and diff presentation modules change; all unrelated current workspace work excluded. Provider/OpenAI SDK file hashes are separately pinned at startup, but full transitive dependency closure remains an outstanding reproducibility improvement. Existing Bonsai/Luna workers and waiting handoff pins are unchanged.
- First launcher attempt encountered a missing new log directory before any inference request. Fixed the isolated supervisor to create log parents, retained original start record, relaunched safely. PID 822917 launcher / 822922 supervisor were live and returning native tool calls at 21:00 UTC. See worker-status/journal for current state; PIDs are historical evidence, not permanent truth.
- Registered the queue/status/plan in shared `check.json.additional_teachers` and journal in `active_journals` without changing the hourly deadline. Main Bonsai/Luna outcome totals do not implicitly include this queue. The default generated snapshot scans all `runs/**/*.result.json`: new accepted explicit-teacher train artifacts enter the next snapshot with all normal admission checks; rejects remain in the inventory ledger. No new snapshot or final dataset published yet.
- Fresh 20:33 manual check found 49 newly finished rows since 20:10, all admitted. Across 17:09–20:33, the separately audited 8 new rejects were 3 wrong booleans, 2 unjustified abstentions, 1 wrong wrapper and 2 exact file-edit preservation errors. All golds supported; no oracle relaxation. One missing newline is an internal blank line, not generic EOF whitespace. Audit: `rejection-audit-0708-other/hourly-audit-1810-through2035.json`.
- Improved `diff_files` model-visible presentation: byte arrays rendered as compact exact JSON-escaped changed text with context, sizes and SHA-256; raw byte-valued ChangeSet is unchanged. Saved problematic previews shrink 20754→1282 and 18703→1735 characters, with identical underlying byte digests. Frozen v25 uses this; existing live runtimes remain frozen.
- Valid `provider_close` cleanup observations lacked role and incorrectly incremented `invalid_events`. Fixed classification in the isolated new supervisor, preserving cleanup as non-progress. Canonical Bonsai/Luna supervisor correction is staged as a candidate only until the partial Luna rolling handoff completes and its exact code pins can be safely revised.

## 2026-10-01 21:40 UTC — DGX Spark deployment, Horizon selected

- User authorized SSH `dgx` over Tailscale, generation on that host, and explicitly stopping the unrelated restarted training. Host `mltick`, ARM64 GB10, driver580.178.04/CUDA13.0. Stopped `sdkb-bgkit` after a30secondgrace and disabled restart (`no`). Exit137: no final checkpoint save confirmed; existing artifacts/mounts preserved. Receipt `runs/dgx-generation-20261001/training-stop/stop-request.json` and remote `~/natlang-remote/training-stop-20261001/`.
- Standard HF caches contained no complete Qwen35B. Direct Qwen3-Coder-Next cache was broken symlinks. An initial Qwen3.6FP8download was started, then **user corrected model choice**: if downloading, use Horizon. Qwen container `natlang-qwen-download` stopped, partials retained, no Qwen generation. Final selection **IFM/K2-Horizon-MoVA-36B-A4B-FP8**, officialApache2checkpoint, pinned revision `feffd71999eb06bfa2fbb8ad220059b694077457`, weights48,357,709,120bytes. BF16alternative74,891,674,376bytes not downloaded.
- User explicitly required HF authentication. Downloader uses `get_token()` from private existing remote HF cache, verifies `HfApi().whoami(token=token)` and passes token explicitly to `snapshot_download`; logs only authentication success, never credential or identity. Initial authenticated Horizon download later replaced by `natlang-horizon-download-fast` with documented `HF_XET_HIGH_PERFORMANCE=1` and8fileworkers, preserving caches. Original stopped containers retained. Token reduces anonymous-request limitations, not a guarantee of network speed. No key copied into repo/CLI/provenance.
- Existing pinned ARM64 image `vllm-node` SHA `666307b207f9f583f6048111e90c5d177355a2f94d9101494c8c85c5b277471d`, vLLM0.29.1rc1.dev467+g0aee727ff.d20260921, has native `K2HorizonForCausalLM`, `K2HorizonToolParser`, `K2HorizonReasoningParser`. Lazy parser dictionaries initially looked empty; explicitly resolving parsers confirmed support. No unsupported blanket model/template conversion.
- Standalone frozen runtime-v26 derives from25onlyadding opt-in JSON `--chat-request-config` sampling/template controls in CLI/provenance and a validated helper. Native tools, prompts, source/golds, admission unchanged. Replaced old incomplete workspace dependency closure with standalone actual-version package/lock; remote `npm ci --ignore-scripts --omit=dev` installed169packages. Root and ARM64Node22.22 independently replayed64nativegoldcases withzero model calls. Root64admittedreferences444linkeddecisions. Remote verified630frozenfiles; manifestSHA `a909719d18a74d61d791508837f311971b4716c042bbcfc11e8fa72e480cc54f`. Current unrelated workspace edits excluded.
- Pilot bundle at local `runs/dgx-generation-20261001/bundle/`, remote `~/natlang-remote/campaign-horizon-20261001/`:64exact existing approved cases,32reducers32workflowcases,16four-casebatchentries. This is **additional teacher deployment comparison**, not64freshcases, and does not reclaim or duplicate ownership of existing local attempts. After pilot, assign new/exclusive work toDGX; actual redistribution of current local queues still requires an explicit unstarted-only handoff.
- Enabled remote user service `natlang-horizon-generation.service` (Lingeralreadyyes). State waiting_for_model_download asof21:37. It waits for authenticated download success, verifies full weight sizes/LFSsha256, validates frozen/runtime/workbundlepins, launches localhost-onlyHorizonvLLMserver, then starts queue. No running Horizon inference/pilot results yet. TP1,32768servercontext,4serversequences,0.65GPUmemoryutilization, dedicated tool/reasoning parsers,8192batchedtokens,prefixcache; teachercontext16384,highreasoning,temp1,top_p.95,top_k20,max8192output. These are initial settings, not a claim of optimal throughput or GB10 numerical compatibility before actual model load.
- Local sync worker pulls remote checkpoints, exported results,journals/logs every45seconds. Assignment staysDGXowned ifSSHdrops; no timed silent reassignment. Raw staging uses `runtime-import-staging/`, excluded by generated-snapshot discovery. Import gate checks exact assigned IRdigest, explicitteacher/model identity and semanticJSON request-controls match, runs current admission, preserves original bytes in content-addressed result paths plus importledger; assignment-mismatched rows held as `.raw.json` outsideautomaticresultdiscovery. Bothadmitted/rejectedvalidassignmentresults retained; DPOeligibilitynotinvented.
- Registered additional teacher, assignment, service, import/status/ledger paths in shared authority underlock, existinghourlydeadlineunchanged. Add remotejournal toactive_journalsonlyafteritexists toavoid falsemissing-readeralerts. Importedeligibletrainresults flowintonextgeneratedsnapshot; no new snapshot/finaltrainingdatasetpublishedbydeployment. Status must distinguish downloaded/serverready/generating/imported/admitted; no success claim from service enabled alone.

### 2026-10-01 21:47 UTC — higher DGX concurrency

User requested substantially more concurrent requests. Updated the initial aggregate cap from 4 to **16**, using four independent collectors with four slots each, including child calls. Server sequence cap is16. The same64cases are partitioned without overlap into four16casequeues, with separate job directories and journals; original4slotplan retained for history. Stopped/restarted only the waiting generation service while changing reviewed pins. Download remained running. Native runtime/source/golds unchanged. The isolated supervisor now reads vLLM's aggregate generation-token counter for server activity; it does not diagnose per-case progress or stuckness. The silence limit remains disabled.

Concurrency receipt `runs/dgx-generation-20261001/concurrency-review.json`; current reviewed plan `bundle/root-approved-plan.json`. Local sync PID848038 initially; check actual status. Bootstrap monitors all four collectors and pauses them if the model server exits. Sync discovers the four journals and preserves remote ownership on connection failures. Import request controls are compared by canonical JSON value, avoiding property-order mismatches. After actual model load, compare admitted decisions/hour and preemption/latency at8/16/32 before claiming an optimum.

Owned script syntax and isolated ARM64 runtime TypeScript checks passed. A full current-workspace type check failed in concurrent, unrelated `src/improvement/teacher.ts` edits; these are excluded from frozen v26. No unit tests were added or run. Horizon download is still in progress, authenticated; no Horizon inference results or training publication yet.

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


### 2026-10-02 08:39 UTC — Obsolete training monitor stopped; Luna rollover released

User withdrawal of compatibility restriction is respected: no restriction added. Remote checksum verification paused safely because the legacy `training_heartbeat.sh` cron wrote two new monitor logs during copy. Preserved both copies, backed up crontab, disabled exactly that old training heartbeat (user already stopped its training), resumed incremental copy/checksum under transient storage unit PID510704. Other cron jobs/model-prep container untouched. No space reclamation claimed before verified cutover receipts.

Luna1 predecessor256 terminal outcomes include one wall-budget partial at case128: provider continued nonempty tool-call deltas for18minutes, no terminal result. Original reviewed controller safely exited without successor launch. Root verified all671 unchanged successor pins and exact failed finish SHA, preserved attempt/no timeout-onlyDPO or positive promotion, approved rollover-plan-v2 SHA14dc6fe4653a29cb13dd50c5c864ef47737b25624a66a65bcee04569b14b0c6d. New controller1159948 launched worker1159952. Detailed tool-stream investigation delegated; resource exhaustion is not proof of semantic error.

Fresh modern static replay1212 exact structural gold matches;1210 current admission/materialization passes, only2 reviewed source quarantines. Qasper300 generation holds are distinct from static eligibility, but sampled traces expose no paper text before gold return: retain candidates and repair visible inputs before publication. Next DGX/Luna2/SpaceBunny backlog preparation is active to avoid idle teachers.


### 2026-10-02 08:54 UTC — Actual finish feature deployed for next generation

Root found frozen30 prompt described eval.finish:true but its older agent/runtime lacked schema+implementation. Current canonical source and compiled runtime already implement it, as user correctly pointed out. Initial prompt-only31 candidate was prepared but NEVER deployed; user requested generation runtime update instead. Frozen32 carries exactly current finish implementation hunks (runtime.ts/agent.ts and compiledJS), inherited30prompt/otherfeatures unchanged. Avoid importing unrelated parallel seed/service/rendering changes; isolated Node TS compilation passes (browser excluded for unrelated missing wllama module). Full630local/remotehashpins verified; manifest472230b75e4bf411a88c958e45e4b419f232dfaa4d58392840d006dd002454e3.

Successor1024 =256reducers+768workflow, no ID/source-group overlap with832priorQwencases. Native32reference replay/admission/materialization1024/5591decisions,0unlinked/providercalls. Root reviewed49localpins,12remotebundlepins, fullremote630closure. Armed controller1174518 / rolloverplanbc0094364f55b287dbfab379e106d7c74871872cfd7487254609910447009e87 waits for complete old512exports+terminaljournal+inactiveunit, then starts runtime32/16requests against existingserver. No snapshot mutation or model restart. Local successor preparations also move to32. Do not claim active old workers already have32.


Runtime32 follow-through08:58UTC: Luna2 successor is actually running worker1177972 on32 after256complete predecessor; SpaceBunny next4 controller1177984 armed32. DGX old512 has exactly1missing export: CommitPack374711bc9d6ebaf124c1 malformedtoolJSONunterminatedstring after2attempts, rawpartial preserved. Exactterminaleventreviewed; noSFT/automaticDPOpromotion. Replaced onlywaitingDGXcontroller(nochildren/successorlogs), v2controller1177045 / plane596f11ff4fc0e4df34c03e77528fe6afee9f241300af9e9295d35511de34a10 expects511exports plus exactmissingID/eventproof. Old512finishing; noactiveattemptinterruption. Bonsai/Luna1v32backlogpreparationcontinues.

Future snapshot prevention: `scripts/freeze_training_runtime.py` now refuses a new snapshot whose compiled prompt advertises finish:true without compiled agent boolean schema and runtime dispatch support. This is a static feature consistency guard, not proof of semantic correctness; existing historical manifests remain reusable. Syntax parsed; no tests run.


### 2026-10-02 09:06 UTC — Runtime32 actually running on DGX

Old512 finished511exports+one exactmissingpartial. Controller v2 safelyfailed because systemd removed completedtransientunit: is-active returns4(absent) ratherthan3(retainedinactive). Fixed genericcontroller to accept3/4 only AFTERcompletejournal/export/authoritychecks; othererrors stillreject. Rootapprovedv3SHA20cbe3ffda3d79030dbdf933164b6f44d8786a096c359e775ccbf7ec7c04e256.

Initialnewbootstrap verifiedruntime/modelpins but fourcollectorsfailed BEFOREmodelrequests: queue source path was root/cases.ir.jsonl butonlybundle/cases.ir.jsonl hadbeenstaged. Rootcopied exactsamepinnedIRtoroot, addedroot-sourceSHApin, archivedfailedstatus/supervisorlog (source-layout-failure*), restartedexistingapprovedbootstrap; noqueue/gold/runtime/modelchange. ActualDGX09:05:24runningbootstrap546911/supervisors546934–546937, runtime32/16requests; sync1184033. LocalLuna2running32worker1177972; SBnext4armed32; Bonsai/Luna1future32preparationcontinues. This supersedes earlier waitingcontrollerstate.


## 2026-10-02 — DGX shared dispatch and256-slot server

User requested applying the linked DGX recipe and much higher concurrent generation. Retained the pinned compatible ARM64 image/model while setting256 sequences,0.65 memory utilization and65536 context; existing FP8 KV/Marlin/prefix cache/chunked-prefill settings retained. Reducing configured maximum context is an operational deployment change; oversized future cases must be reviewed, never truncated to fit. Explicit vLLM entrypoint fixed a startup-only container failure. Old server retained stopped.

Safely drained1024 assignment after432 exact outputs (408admitted/24rejected); transferred only592 never-started rows into37 unchanged batches with fresh output paths. Replaced four fixed serial supervisors with a drainable shared dispatcher,128→256 requests after two clean-accounting completions. New gate fixes prevent failed finishes advancing ramp, cooldown bypass, and missing/nonzero child terminals silently burning the queue. Status/journal paths remain visible to importer; local import roster has592 rows while remote execution source retains full original1024 indexes. Actual runtime dependency files are also pinned; no historical runtime mutation or raw data loss.

Full32/64/128/256 benchmark completed without failures/OOM; captured output throughput337→459→557→655tok/s, short synthetic444→540→635→699. Prefix reuse/four-context fixture and capped reasoning output limit interpretation; actual production admitted cases/hour remains the required follow-up. The claimed2835tok/s recipe figure was not reproduced. Larger reviewed backlog is being prepared; no correctness/admission checks relaxed.


### First production completion and recovery10:05UTC

Initial shared dispatcher crashed because completion handling looked up key one level too shallow in its active-task wrapper. Parent-death signaling stopped its children;38saved results remained. Separate output-accounting Node helper failed EAGAIN reading a large nonblocking stdin pipe after module imports, falsely labeling16 exact saved/exported SciFact IR records invalid. Digest helper now reads a temporary regular file; exact complete accounting revalidated all16 without generation. First completion was an infrastructure error, not a model negative. Historical failure journal retained alongside root-recovery-accounting-v2 receipt. Restarted distinct pinned dispatcher/supervisor-v2 service592409 at10:05, reusing compatible saved results. Canonical helper failure classification is now accounting_error; next canonical dispatcher deployment marks paused on unexpected exception rather than leaving running status stale.


### Reviewed successor continuity and source quality10:26UTC

Staged2048fresh cases,538reducers26.27%,1510primitives,12035native/current-admitted reference decisions. Root automatic successor plan is armed behind complete592-current-case imports and exact batch accounting; no GPU lifecycle interruption. Thirty-two16-case collectors with8request caps can expose512 active cases but stay under256 total model requests, smoothing batch-tail utilization. Added multi-journal predecessor verification instead of treating the shared scheduler as four fixed shards. Conditional launch rehashes full runtime dependencies and validates all37 predecessor exports; recovered16-job accounting is explicit, not an exception hidden in a success status.

Rejection sweep found six correct saved files with malformed final strings. Added generic canonical prompt clarification, awaiting a new immutable runtime deployment; no parser/schema relaxation. Source-unit conflict for TATQA13d32af1-1901-4340-b9b8-023ed52ec0ec is now centrally held pending adjudication, including current compiled policy;48/million gold preserved, no DPO negative. Other reasoning mistakes remain rejects.

Restarted authorized disk relocation recovery at low CPU/IO priority with2GiB MemoryHigh; guard refuses nested mounts before exact-backup deletion. It verifies the retained backup as a byte-identical subset of external copy, releases only that verified backup via a single Docker bind, then moves the second authorized tree using checksum-before-cutover. `.cache` stays; no completion/space-reclamation claim until receipts.


## 2026-10-02 11:19 hourly sweep

Actual50-minute tool sleep completed. Reviewed successor automatic handoff succeeded after592exactimports (573admitted/19rejected). Next2048has1020saved/927admitted/93rejected at11:19;32slots×8requests configured,56clean batches/no failed batch completions at11:23. New rejection audit delegated; no semantic gates relaxed.

Storage subset validation had two operational false-alarm causes: repeated itemization (`-i` plus long flag) lists unchanged files; backup-directory timestamp changes after partial cleanup are not byte-content differences. Corrected to one itemization, excluding only directory mtimes for retained subset verification; exact full checksum and regular-file/link checks remain. Fresh checksum restarted before any deletion. Cheap metadata-only corrected check emitted no differences; not a substitute for checksum. Luna2 had actually finished rather thanfailed; nextqueue supply was missing/staleauthority and is being repaired.

## 2026-10-02 11:47 UTC — worker continuity and fail-closed liveness

Recovered Luna2 after its prior256queue completed and authority remained stale; fresh187case/317native-decision queue is genuinely running1278024. SpaceBunny next4 source-review pin failure was re-reviewed before relaunch; all256cases/1599decisions valid under current admission, versioned supervisor retains old bytes and fixes large JSON stdin. Startup errors now publish paused_setup_failure only under acquired campaign lock. Shared successor liveness uses actual non-zombie PID rather than one hardcoded supervisor filename; this covers versioned supervisors and provider launchers. Reused live PID conservatively waits. No admission relaxation. Python syntax parsed, no unit tests.

Qwen2048 audit1303imports/1205admitted/98wrong_return found zero reasoning-channel contamination. Individual source-backed MuSiQue variants and ten source/oracle disputes are under review; do not use broad substring/fuzzy matching or make disputed-source negatives. Minimal immutable33 JSON-return prompt candidate staged but not deployed. Automatic teacher successors and nextDGXbacklog preparation continue before next50minute active sleep.

## 2026-10-02 11:49 UTC — verified first backup release

Corrected storage checksum returned zero differences; sdkb-runs backup removed11:48:19 with durable exact-path release receipt, path-preserving external symlink verified. Internal47→202GiBfree. Second146GiB bgkit-data-nvme copy now underway; original retained until full checksum/cutover. `.cache` unchanged. No generation interruption.

## 2026-10-02 12:00 UTC — shared pool architecture and export serialization

Investigating process-reserved capacity: live107requests/0waiting while28batchleasespending, so merelyraisingperbatchcaps is insufficient. Preparingonecollectorpool withshared256requestgate/320caseworkers, retainingcaseindices/sourceprovenance/freshuniformseed andpercase40turn384requestbudgets. Active32campaign unchanged. Exportcoordinator removesconcurrentaggregate snapshot race: durablejobsfirst, serialized100mscoalescedmerge, finalflushawaited, stopnewcaseadmission onexporterror. Native8casesoutofordercompletedandallmergedorder/provenance/admission/materializationpassed;0modelcalls/no unit tests. Separatepoolrunner/drainhookunderimplementation, notyetdeployed.

Rejected draft aliasregistry accepting fullprose answers despite answer-only taskcontract. Standaloneexactsourcevariants only; rawproseremainformatrejects. Apparentjazgoldtypo heldpending versionedsource-backedcorrection, notstaticpositive merelyviajazzalternate. This isreviewofunpublisheddrafts, no existingrawartifactwasrewritten.

## 2026-10-02 12:03 UTC — current source admission holds

Root approved ten exact source-pinned MuSiQue holds after reviewing source/question/gold conflicts and matchingreceipt. ID-only pendinglookup stillenumeratesreview; actualrecords requireexactrevision/snapshot/fullprompt/gold, so separatelycorrectedcontracts arenotblanketheld. LadyGaga2017modifier ambiguous,notdefinite contradiction. jaztypoheld topreventstaticbadlabel. Raw/historypreserved/noDPOnegatives. Frozen32/33unchanged; futurequeues currentpolicyrechecked. Five standalonealiases stillunderfinalproofreview; explanatoryprose remainsrejected.

## 2026-10-02 12:09 UTC — narrow answer forms and independent slot controllers

Five exactsource-backed standaloneanswer forms approved;5nativeaccepted/4capturedprose rejected/1rawfulldateaccepted/5idempotent. Capturedteacherhashes separatedfromacceptedanswerhashes. Builderintegration versioned, nooldraw/frozenrewrite. WeakNiger/Jaznotaliases. Sourceholdreceiptrootapproved. SingleorpairedLuna controllerplans supportedwithdistinctnumber1/2monitorbindings, preservingauthorityslot/terminal/hashguards. Repairrepeatsofrejectedsource-validcases explicitlyallowed; nofakefreshness/noautomaticcrosspromptDPO. Ninecase repairdrafttoo short, largerbacklogpreparation requested. Local33 sourcehash630doesnotprove mutableworkspace dependencies; runtime34poolmustincludeactualimmutabledependencyclosure.

## 2026-10-02 12:44 UTC — central pool and empty provider response recovery

DGX next2048 completed2048 exactimports:1937admitted/111rejected. New1024selection includes300newworkflowpair-directoryreducers(29.3%);sourcequestion reuse acrossprimitive/reducer shapesexplicit, Qasper heldsources excluded. Replace process-reserved batches with one320-case collector pool sharing256request permits, immediate per-case replacement, durable per-case files/final-only aggregate. Defaultwalltime disabled; case40turn/384requestlimits retained. Exactruntime/config/native authorization required; sealeddependencyclosure and distinct memory guard reviewed beforelaunch.

Eight SpaceBunny empty-response failures contain zero replies/deltas; preserve oldpartials/raw/journals, new8case recovery, then256successor. Exact SDK emptyresponse error now operationalretry under existing exponentialbackoff/jitter, not wrong_return orDPOnegative. TypeScript noEmit passed. Bonsai automatic256successor armed1324745; independentLuna2actual33handoff1317866. No changes to frozen33 or oldcasegold.

## 2026-10-02 12:59 UTC — 256 real requests and continuous case refill deployed

Deployedone320-caseDGXpool/actualshared256gate; observed256runningrequests,38%KV,27GiBMemAvailable. All1024native/currentadmit/materializer13,590decisionspassed. Corrected300 malformedtypeannotationstringswithnewversionedIR IDs, retainedoriginalgolds anddrafts. Wrong same-name481file runtimebase caughtbeforelaunch; correct630base +privateclosure sealed34r4. AddeddiskANDmemoryguard, fsyncedcasejournals, final-onlyaggregate, gracefuladmissiondrain/defaultwalloff. ExistingNode18 interpreter passesremote3nativefixtures butenginesmismatch recorded; abortlistenerwarningauditpending.

SpaceBunny all8 provider-empty responses recoveredexactly; next256running. FixedOpenRouterhandoff race: absentinitialstatus waitswhileauthoritative launcherPIDalive; exitedwithoutstatusraisesreviewerror. Newrootplans/launchrecordsv2preservefailedcontrollerproof; nofinishguardbypass. Old248outputs+8incomplete explicitlyarchived/transferred. No unit tests; typecheck/syntax/nativequality proofs only.

## 2026-10-02 13:24 UTC — explicit missing-result semantics and lifecycle cleanup

Root rejectedbothnewsourceholds proposedindeltareview: sticker/kitmismatchisintentionalfalseproposition; missingSOWhasexplicitfalsefallback. Howevergenericblock-if-missing guidanceconflictswiththatfallback. Fixedcanonicalprompt tohonortaskdefinednegative/no-matchresults andtreatdeclarativebooleanclaimsasevaluationtargets. Sourcegoldunchanged/frozenruntimeunchanged; deploymentnext36pending. Don'tlabelprompt-conflictedblockedcaseDPOnegativewithoutcausalreview.

Futureclose lifecycle abortscompletedprivate signalafterdrain; futurereaperretainshandleandwaitsforTERM/KILL+reapbeforelockrelease. EightnativefixturespasslocalNode24; highconcurrencywarningreproductionnotcovered. Fulltscformattingchangedcompiledpolicyhashandsilentlyinvalidatedwaitingpins; restoredreviewedcompiledbytesforsameunchangedTSsource, documentedrootpinreview. Useownedfiletranspilesonly.

Luna227validfreshcasesproceededafter256quantityminimumremoved;300heldQasper/sourcependingexcluded. Nativepreflightbare-read refsarenotpublicationpositiveswithoutvisible-inputproof. Actualteachergenerationseparate. SpaceBunny256allaccounted,next300newpair-reducercandidateunderreview. Secondstoragechecksum/cutover/releasecomplete13:15;internal346GiBfree,.cacheuntouched.

# High-concurrency production check and Luna handoff recovery — 2026-10-02 13:44 UTC

- DGX central pool still healthy: 253 active requests/1 waiting, GPU96%, KV56.8%, zero preemptions, 26,720MiB available RAM. Latest128 imported artifacts126admitted/2wrong_return rejects. New invoice rejection asks whether document is NOT an invoice; observed document explicitly INVOICE, model returnedtrue ratherthanfalse. Supported gold, no format/parser defect found. Prior customer pair also reversed boolean criteria. Raw failures preserved. Production sample475outputtokens/s remains short-interval evidence, not full-campaign benchmark.
- Luna1 predecessor256finished. Its automaticcontroller1308753 had exited during transient full-TypeScript-emit source-review.js byte mismatch; restoring reviewedbytes did not resurrect it. Root reverified all659pins/exactpredecessoraccounting and created new recoveryplan/record, preserving old crash. Newcontroller1376313 successfully launched actual Luna1worker1376317 with256reviewed33cases. RecoveryplanSHA1ad6f74202e806959f93d1d77583c4fff30f6b94e3c0f131dc5c4960a0834856. Bonsai controller1324745 still alive; no duplicate workers.
- Luna2 current27finishedexact27/27; next314beingprepared (14unattemptedsource-reviewedrepairs +300directory-v7cross-teacher cases). Three previously successful exactLunacasesremoved; fiveTATQAfoundonlyinretired-before-launchqueues mayproceed. No successorlaunchclaimyet.
- Root caught wrong predecessor in SpaceBunny300draft: it audited oldspace-bunny-reducer-successor, not actual finishedspace-bunny-workflow-next5-v35. Old188/68stats MUST NOT be attributed to current256. Correctedpacket/predecessorreceipt pending. Root independently checked all630runtime36files, exact35base outside5reviewedprompt/runtime paths, and actual300native rows matchingselectedIR; allaccepted,6311turns/6011approveddecisions. Root-native-check underbundle. Existing drafts preserved, no launchuntilcorrectactualbinding/allruntimepins.
- Source agent preparingcorrectedBunnypacket; qualityagentpreparingLuna314; rejectionagentpreparingnextDGXbacklog and reproduciblev2→v3annotationtransform. No live34mutation, no unreviewedtrainingpublication. Nextactual50minutesleep follows urgentworkerreplenishment.

---


# Space Bunny300 launched and Luna history correction — 2026-10-02 13:50 UTC

- Actual newSpaceBunny300directory-v7 worker isrunning: launcher1379473/supervisor1379568, immutable36. Rootreviewall630runtimefilehashes/659preparedpins/3284providerdependencyfiles and exact256actualpredecessor result/output/IR/trace comparisons. ActualrootrolloverSHAfa406adb4c83f39f08916a2bdb292bd58ff15509e0dfc685f6f0527034a1a1b3; workerSHAe8c7420a452f4f76bff20ca579c89ab56bf11d282688dc96ccee852d29c7a260;665rolloverpins. Firstnewcaseaccepted,8turns; capturedcontextcontainsnewclaim-to-evaluateprompt, provenancefreeStealthonly/no distillationflag. Currentprior256was255accepted/1reject, notolder188/68. Actualpredecessorplanv35-v2 used inrootrollover (agentdraftnamedv1). SourceID/programoverlap0;9sourcegrouplabelsoverlapexplicit. Preservedwrongdrafts review-history. Local36doesnotsealallNodepackages; knownparent-dependencylimitation recorded, verifiedprovider3284pins only.
- BroaderactualsavedLunahistory supersedesearlier14/17nonrecentclaims: all12workflowrepaircandidates hadpriorLunaresults (11accepted,1rejected); fiveTATQAonlyappearinretired-before-launchqueue/noresult. NextLuna2candidate now306=5unattemptedTATQA+1explicitpriorfailedworkflowretry+300cross-teacherpairreducers. Rootauthorizedthisshape; native/rootapproval/launch stillpending. No falsefreshnessclaimforretry/sourcequestionreuse.
- NextDGXselection conservativecurrenthistory811fresh/unheld (356reducers43.9%), plus42nonoverlapreviewedrepairretriescandidate=853. Other25repairIDs overlapcurrentlocalqueuesandexcluded. No2048quantityquota; quality/sourcecoverage wins, continuouslysealedbacklog. Nativeproofnext/policies pending, selectioncounts mayshrinkonproof. Sourceagentpreparingcompleteclosure37fromexact36andisolatedsupportedARMNode, no live34/serverchanges.

---


# Worker supply restored; next active50-minute check — 2026-10-02 13:58 UTC

- Actual Luna2worker1383567 nowrunning306 reviewed36cases; controller1383562/rootplanSHA77de00963d722fdfd265bf18703304f345f77a4eaefb56e86a735c4c3b5d35c3; all659pins/native306IRexact/allaccepted/6047decisions verified. Runtime/program/goldsunchanged. Agent's per-rowselectionproof defaultedbothpriorLunaID/sourceoverlaptofalse; aggregatereportcorrectlyrecords1explicitretry/596sharedpairsourceIDs/5sharedTATQAsources. Rootlaunchedbeforeagent'slatewarningarrived. Metadata correction requiredbeforepublication/inventoryuse: agentpreparingseparatev2proof, originalapprovedpins/rawworkerpreserved. Do notinterpretoldproofasfreshsourceclaim orautomaticcausalDPOpair.
- Allteacherscurrentlysupplied: DGX256requestgate/320activecases; Bonsai778887 long513queue+armedreviewed256successor; Luna1actual1376317/new256queue; Luna2actual1383567/new306queue; SpaceBunnyactual1379568/new300queue. Verifiedlatest-worker-supply.jsonreceipt.
- Production70.59secondssample333.46outputtokens/s,6736prompttokens/s countedbyvLLM (promptcounterincludescacheuse; notphysicalprefillthroughput). Earlier110seconds475.23outputtokens/s. Workloadvarying; do notclaim475sustained/fullcampaign. KV57.2%,251active atnewsample, no reportedresourcefailure. Completedartifactmixstillmostlyprimitives, reducerqualityconclusionspending.
- DGXnextbacklogconservative723candidate=687freshunheld+36explicitreviewedrepairretries,376reducers52%; countssupersedeearlier853afteradditionalactual/plannedLunacohorthistoryfilters. Stillpreparingnativeproof, notqueued/deployed. Future37completeclosure+supportedARMNode preparationonly, live34unchanged.
- Startingactual50minuteclock.sleep now; absoluteUTCdeadline **2026-10-02 14:48:42 UTC** persistedruns/generation-check-20261002/active-sleep.json. Handleurgentnotificationsthenresume SAMEdeadline. No independentagentwakeschedulerclaim. Onwake checkservices/imports/resourcepressure/newrejections/workerqueues/correctionproof/nextqueue.

---


# Metadata proof corrected; actual sleep starts — 2026-10-02 14:00 UTC

Root independently checked all306correctedselectionrows againstoriginals: onlyhistoryflags/membershipmetadata changed, all3,584referencedmembershipfiles agree withactualsavedIR. Exactlyonepriorprogramoverlap (authorizedfailedretry); all306havehistoricalsource/groupoverlap. Correctedv2proofSHA3bcb8f8482c7b00e4a2e803b2bb17ea887e8b55bfb95c2202b8a524ff37933e0; rootaddendumSHAdb232c760e3e3acdcdf4892dfea26fbf87e99a14bf18726cf2debc14c79717c2 nowboundactualLuna2authorityworker/launchrecord. Originalpinnedproof/IR/runtime/queue preserved. No publicationcausalDPOclaim. Prior13:58sleepentrywaspreparedbeforeurgentcorrectionarrived; ACTUAL50minuteclock.sleepstartsnowdeadline **2026-10-02 14:50:37 UTC**, active-sleep.jsonupdated. Resumeabsolutedeadlineafterurgentinterruptions.

---



## 2026-10-02 15:07 UTC — inventory scope and infrastructure corrections

Full-training totals must include static, deterministic reference, converted historical and all model-generated teacher lanes. Recipe-v31's Oct1 snapshot is stale; current inventory is isolated from the default recipe and does not publish training-ready data. Distinguish unique programs, unique trajectories, approved decision records and final rendered tokens; never subtract completed batch counts from assigned case counts. Native reference preflight rows are not model-generated outputs.

Luna1's approved successor failed before model request because its output log parent directory was absent. Reverified all659pins, created the missing parents and resumed identical pinned queue/runtime/journal; actual1435328, first complete model trajectory observed15:05. Preserve raw start/crash log. Future launcher preflight must validate/create every output parent, and worker health must require live process/progress rather than a stale launch receipt. Do not mutate supervisor bytes pinned by armed controllers.

Fledge on Zen returned permanent403 FreeTierError country restriction on five zero-reply attempts. Both workers and parent have stopped; no dataset-positive or DPO-negative interpretation. Preserve reviewed plan/runtime/keys ownership and raw errors. Future versioned supervisor should circuit-break clear permanent provider eligibility errors on first observation. Provider eligibility resolution required before relaunch; no paid fallback.


## 2026-10-02 15:27 UTC — Bonsai-only stop and local training pipeline tests

User corrected earlier all-generation transfer: only Bonsai stops, Luna/Bunnycontinuehere;DGXQwencontinues. Stop watcherbeforeDockerbecausewatcherrestoresmissingcontainer. Durablegeneration-disablemarker applieswatcher/serverstartup, twoBonsai-onlyrecoveryapprovalsretired. UnfinishedBonsai work transfersasfreshQwenattempts withunchangedIR/golds;partialcross-modelcontinuationforbidden. LocalGPU nowavailable foruser-authorizedpipeline tests/fixes. ExistingDockertrainingenvironment chosenbecausevenvlacksTorch. Currentstaticinventorycorrected32,615approvedvs32,952rawlabelcount;heldrowsnottrainingpositives.

## 2026-10-02 — Template boundaries and full-data memory use

- Resolve the assistant terminator from the actual tokenizer chat template, including registered added tokens, rather than assuming generic EOS. Explicit terminator overrides must agree with the template. All-invalid render attempts fail after preserving evidence; required pipeline renders cannot commit empty corpora.
- Full current-data builds must use memory-bounded preparation/rendering on this host; measured workflow input alone projects ~6.3 GiB decoded RAM. Preserve source groups, dedup and deterministic split behavior while changing storage strategy.
- DGX successor gate must read actual final runner accounting (final status lacks progress/collector_pid) and wait through normal final-status/process-exit/lock-release races. No concurrent second request gate.

- Training preparation, assembly, rendering, and audit now have opt-in memory-bounded modes with source-group/dedup/order parity. Disk guards fail before creating large scratch outputs; full-production-build readiness remains explicit rather than inferred from successful representative smoke.
- Qasper static exploration must expose read evidence in tool output. Repair scripted references using attributed original human evidence; preserve old trajectories, golds and holds, version changed actions, and do not describe oracle-selected navigation as generated model reasoning.

- Root campaign completion audit corrected Space Bunny300-entry queue to299results+oneincomplete. Lastjob/processcompletion is insufficient: inspect every entry's exact output accounting. Retain interrupted model traces as infrastructure/runtime evidence, not answer failures or preference negatives.


## Full student-training scope correction

The 500-step local LFM run was a bounded experiment chosen by the agent, not a user limit. It sees4000examples from a7993-row subset and must not be represented as full training. User asks for full training: prepare the complete v13 corpus for the LFM tokenizer/audit while the GPU keeps training, then run a full audited epoch with recoverable checkpoints and evaluate heldout results to select further epochs/checkpoints. Preserve existing run identities and receipts; no silent scheduler or corpus mutation.


## Muon and full-state resumability

User requests Muon preference and emergency checkpoint/automaticresumption likebgkit. Newtrainingphase uses nativeTorchMuon forhidden2Dtrainablematrices with match_rms_adamw LR scaling and auxiliaryAdamW forremainingparameters. Serialize both completeoptimizers plus parameterpartitionidentity; mismatches failclosed. ExistingAdamWrun/optimizer history staysintact; newMuonphase intentionally startsfreshoptimizerstate fromadapterweights. SIGTERM/Ctrl-C checkpointatcompleteoptimizerstep; hardkill/powerfailure recoverslatestcompletedperiodiccheckpoint. Durablelauncher implementation/test remainsinprogress untilverified.


## Full-corpus initialization correction

Source-group audit found two groups in the 309-row pilot training split also belong to full-v13 heldout (`inline-curriculum:s5:child_sufficiency:refund0:undetermined`, `s5:child_sufficiency:refund0`). All subsequent pilot-derived adapters inherit that exposure. Full-v13 Muon training therefore starts from the pinned base with freshoptimizerstate; keep protectedheldout unchanged and preservepilotadapters for their originalscoped evaluations. This is a necessary evaluation-integrity correction, not a requirement that fullcorpus training alwaysstartfresh.


## Teacher overlap and actual full-run launch

Cross-teacher source-group sharing alone is not a data-quality fault: distinctquestions inoneunderlyingcase and deliberatealternative teachertrajectories may be useful. Keep same-teacheractive/positive dedup, currentnative/sourceholds, explicitoverlap provenance and protectedtrain/heldoutclosure; do notinventglobalcross-teacherdisjointness blockingmeaningfulwork. FullLFMv13Muonrank16LoRAepoch launched underpinnedv3serviceafter root26pin+imagecheck andactualaudit/mix/inventorypreflight. Oldpilotgracefullystoppedstep351; fullbaseline2.2647225933 andfirst10steploss1.9029508 finite. First20stepfulloptimizercheckpointpending; reportactualreadinessonlyafterverified.


## 2026-10-03 first 50-minute cadence check

- DGX1024 produced 1,023 exact exports; case717 hit the 384-request collection budget. Preserve its partial/error and incomplete batch status. The existing gate correctly blocks, but future queue handoffs must support explicitly reviewed terminal exceptions so one resource-limited case does not idle the GPU. Separate repair remains required; no positive/DPO-negative relabeling from resource failure.
- Same-Luna accepted/active suppression leaves seven cases in the 951-source pool; approved and launched those seven rather than padding. Cross-teacher coverage remains allowed.
- Six answer-only rejections across Luna2 workflow8 and Bunny395 are under evidence review; no blanket numerical tolerance or broad prompt change warranted by the initial review.
- Supervisor status now reads current checkpoint progress and validates identity (be20d7d); lifecycle status files stay immutable to this observation, and the active pinned trainer/supervisor is unchanged.


## 2026-10-03 generation recovery and prompt clarification

- Restored the same951-case DGX queue with canonical sync paths. Preserve original v38 argv/authorization and v39/v40 pre-provider failures. No missing result was relabeled complete. Use explicit review of terminal resource exceptions to move on to independent queued work.
- Actual runner preflight and guard main path under the shared lock are required: function-only guard checks missed a self-lock check, and native case proof did not validate launcher journal basenames. Stage a reusable preflight-only runner and canonical path builder to eliminate these manual inconsistencies.
- Canonical jobs/status paths also fix a potential importer gap for versioned directories. Local importer uses frozen x64 current598 only for digest/current admission; actual teacher execution remains ARMv38, recorded separately.
- Commit6cc0bf6 clarifies that finish:true ends the call with its fresh final expression/explicit return, and each requested predicate must be represented. No runtime semantics or active frozen prompts changed. TreeDST717 spiraled on unsupported walkers; current direct-path tree-edit guidance already addresses that strategy. No blanket timeout or broad delegation rule added.
- Luna7 geographic full-sentence response may be a migratable format failure rather than a semantic error; review source/question before any transformation and preserve the raw attempt. Do not classify it as a clean semantic DPO negative or silently expand aliases from rejected output.

## 2026-10-03 queue handoff and runtime capability fixes

- The first phase3 Luna handoff changed a mutable state file that was byte-pinned in its own two-slot plan, blocking slot2. Preserved the partial handoff; launched only the unstarted slot through a separately approved plan. Commit336e3ee rejects authority/state pins, including symlink aliases, before a handoff claim.
- ARM38 silently ignores case-events-file, final-export-only and drain-file. Keep the healthy active queue intact, label derived inventories honestly, and inspect actual supported shutdown semantics. Future runner launches require capability checks; source-presence checks detect these omissions but are not a complete behavioral proof.
- Commit336e3ee adds canonical shared path validation and a preflight-only mode that writes nothing even on failure. Authorization now pins the path helper dependency as well as the runner. Nine focused CPU tests passed. Future queues need new reviewed authorization; active frozen runtimes remain unchanged.

- Phase4 v1 never launched: full-v13 source closure caught protected test identities despite original workflow IR train labels. V2 excluded9 unsafe candidates; all192 selected tasks passed native proof and protected exact/source/group closure. Approved v2 successor plans preserve the blocked drafts. Dataset revision sharing alone is not a split leak.

## 2026-10-03 01:43 UTC cadence sweep

- Main training checkpoint940/7,520 examples and later1000, zero skips, GPU100%; about38.7h remaining from recent100-step throughput. DGX951 finished01:37:51 with951 exact exports,0errors. Raw outcomes924accepted27rejected; importer admitted951 means assignment/source-policy import eligibility, not951 positive SFT cases. Bunny phase4v2 finished64/64; Luna successors remain active. New queues and modern ARM runtime are being prepared.
- Independent phase3 review covered31 rejected outputs; no transport/parser/runtime failure established. TatQA281ec4c8-1632-40e9-bdf2-8ceac4303d61 is now source-held (commit6d7d277): reference incorrectly uses fourth-quarterEPS1.71 plus second-quartereffect0.02; second-quarter table1.08 and note0.02 support1.10. Preserve raw/gold; no clean DPO negative. Active corpus includes8 source-linked decisions; actual source exposure receipt proves only table-read row887 consumed at the check, wrong answer steps still future. Prepare an explicit exclusion/resume transition before remaining source rows; do not silently mutate frozen input.
- Commitd68321a adds source-safe first-append preparation/consumer and composite-identity protected TEST packet tooling. Eight intake/order tests, one AdamW continuity test, four TEST packet tests and one actual nativeMuon+AdamW auxiliary CPU serialize/LR-anchor test pass (Muon in active training image with CUDA hidden). Failed source-gate checks now occur before combined corpus writes. Active frozen trainer is unchanged; production chained append and quality-exclusion transitions still need implementation/verification.
- Bunny reducer raw outcomes16/64pass,48answerreject, mainly unknown-record booleans encoded as strings. Legacy directory-v1 IR declaresRecord<string,unknown>; newer workflow adapter already declares fields from visible question schemas. Hold the next Bunny98 legacy draft until a versioned typed-field IR transform/native proof is reviewed. No blanket string-to-boolean coercion or gold rewriting.

## 2026-10-03 02:25 UTC — modern generation and typed output recovery

Moved the new DGX102 pool to current sealed ARM runtime444f5047 after zero-provider native/source/CLI/behavior review; x64 import57042f53 is separately pinned. Preserve completed951 and incomplete1024 histories. Import eligibility is not positive SFT acceptance.

Typed98 workflow outputs now have source-visible primitive field contracts; gold/files unchanged, delegation optional. Root prevented mislabeled pin maps, then recovered an unstarted queue after the rollover dispatched itself and failed argparse before provider calls. Original failed receipt is preserved; never reset the fence. Correct actual worker status lives beside its plan. Rebuilt authority binding from current plan instead of retaining predecessor counts/proofs; archived prior metadata. Generic rollover checks will prevent these launch errors before claim.

Qwen951 rejections:16 semantic answer errors,2 byte-exact file mismatches,2 precision candidates and7 explicit blocked traversal cases. Review is saved under rejection-review-v1; do not silently broaden admission or label ambiguous/resource failures as semantic DPO negatives.

## 2026-10-03 — recovered SCONE contract visibility

New DGX102 batch finished78 raw accepted/24 rejected. Several SCONE failures omitted empty/unaffected beakers. Inspection found the root instruction lacked the encoding/action rules, which existed only in the optional transition function. Future static state adapters now include the same visible transition context in the root, and explicitly require all seven ordered beaker positions including empty/unaffected ones. This changes future prompt contracts only; source transitions, independent oracle and expected states are unchanged. Existing frozen IR/runtime/training remain unchanged. No acceptance broadening or compulsory delegation. Syntax check passed; no model comparison performed yet.

## 2026-10-03 — flattened connection errors and terminal accounting

Bunny typed98 completed96 exact exports, with two `Connection error.` transport failures. The original raw partial/error/journal bytes remain preserved. A copied-partial retry queue for only indices66/93 is root-approved, separate journal and accurate two-case authority binding. The649 successor guard refused the incomplete98 predecessor; failed waiting receipt remains, with a separately reviewed new handoff behind the2case repair.

Actual transport retry classifier missed the SDK's exact flattened `Connection error.` message. Added a narrow exact-message match to the future runtime retry policy; existing exponential backoff/jitter/caps remain. Three focused retry tests passed against TypeScript source without rebuilding or mutating active frozen runtimes. This is not deployed to currently pinned workers yet. Worker launcher reporting `finished` on supervisor exit0 despite incomplete accounting remains a separate bug to fix for future launches; queue guards correctly require exact output accounting.

## 2026-10-03 02:53 UTC — refills and proof metadata

Root launched two Luna slots212/211; task-levelpayload freshness screen scanned8227 priorLuna IRs. Bunny2 copied-partial transport repairs completed and649 deliberate second-teacher trajectories started. DGX20 historical recovery launched after exact current native and full-v13 program/source/group protected-test closure. Root rebuilt assignment/authorization metadata instead of carrying old102 hashes/proofs/counts or ARM import runtime; original drafts preserved. Local importer required bundle/chat-request-config.json, added exact already-approved bytes after ENOENT; import then succeeded.

Keep prompt-contract ambiguity out of clean semantic DPO negatives. New directory/composite tasks can be valuable despite accepted standalone components; strict component-positive suppression applied conservatively to2 repairs but is not a global exclusion rule. Preserve distinct task semantics/source lineage and actual whole-task history.


## 2026-10-03 03:43 UTC — source holds, continuation safety and replenishment

- Finished Luna423 yielded422 raw accepted/1 rejected; Bunny649 yielded640/9. Ten reviewed answer failures are semantic or answer-only format failures, not infrastructure defects. Keep their raw rejects and explicit recovery identities; no fuzzy aliases or blanket admission changes.
- DGX20 finished12/20 raw accepted; started56 current native/source-safe directory reducers with56workers/concurrency256 and existing high-throughput request settings. Reviewed whole-task history/protected source closure; primitive component overlap alone does not suppress distinct composite tasks.
- Corrected architecture proof labeling: local x86 execution of packaged ARM JavaScript is not actual ARM native proof. Preserve the bad-label correction and real DGX execution proof; distinguish native case reports from trajectory rows.
- Exact Bugabula MuSiQue source hold added02ff4cc because1400m is species elevation, not country highest-point evidence. Preserve original gold/raw and exclude its one future training row alongside8 TatQA source rows. Do not invent replacement answers or semantic DPO negatives.
- Root blocked exclusion candidate v3 before activation: exact parent checkpoint hashes apply only at the transition boundary; later resumes must accept legitimately advanced optimizer/model/RNG bytes while checking transition/corpus/order/progress. All future resume policy/code artifacts must be frozen, not mutable repository paths. Preserve original cosine horizon13165 and its intentionally unused one-step final tail. Apply only from a freshly quiesced checkpoint with actual consumed-row counts.
- Future OpenRouter launcher73256bd now requires complete explicit queue accounting before reporting finished. Validated against actual649/649 and96/98 historical journals without changing old evidence. New plans pin the new launcher; existing one-shot receipts are not reset.


## 2026-10-03 04:24 UTC — applied exclusions, real continuation and launch corrections

- Applied exact9-row source exclusion at original durable1686/13488; preserve3 historical harmless reads, remove6 future rows from example target. Target105311, rebasedcursor13485, unchanged13165-step LR horizon/184Muon state/RNG/weights. Original artifacts preserved; clean train105308/heldout5410/reducer31.5256%.
- Plan/runtime review prevented bad policy mounts, old supervisor CLI and mutable policy/code dependencies. Production readiness rejected frozen policy alias because the original report pins its canonical path; narrow alias verification preserves original ready/report/hash identity and exact sealed bytes. No global gate relaxation.
- First production continuation exposed an additional `held` local shadowing the heldout list. Stopped at1689 after3 valid steps, then copied that checkpoint to source-clean-v2 with fixed trainer/5410-row guard. This preserves useful compute rather than rolling back to1686. Original and v1 services disabled/operator latched; activev2 enabled for reboot resume. CPU optimizer proof alone did not cover full main-function split integrity.
- Luna fresh600 and Bunny fresh256 launched after root exact native/IR/payload joins and protected closure. Corrected Bunny bad semantic pin map/status-path draft in separatev41; generic validator passes. Tasks single-call workflow coverage, not directory cases.
- DGX56 exported all56,54 raw accepted; Qwen20 file failures replayed correctly on actualARM/x64 and are model edit-string errors. Quiesced/no-answer remains coverage failure, not semantic preference negative. Larger Qwen pool pending; do not silently turn import eligibility into positive outcomes.

## Latest continuation — 2026-10-03 04:54 UTC (ongoing)

Main source-clean-v2 training advanced to step1830 with no overlength skips. Root independently validated the actual step1700 checkpoint for later resume beyond the exclusion boundary; weights/optimizer/RNG changed appropriately while the approved corpus/exclusion marker and cursor relationship remained valid. Receipt `runs/monitor-cadence-20261003/source-clean-live-resume-review.json`.

DGX56 finished56 exports (54 raw accepted/2 semantic rejects). DGX now runs213 source-reviewed nonworld cases: service `natlang-qwen36-current-train-refresh-20261003-v6.service`, local `runs/dgx-qwen36-current-train-refresh-20261003/pool-v6`, remote `/home/werg/natlang-remote/campaign-qwen36-current-train-refresh-20261003-v6`. Root assignmentSHA `03eead1aa8f6ca4eb2a90f0c296d98761b95f8f3ca7fa72b7b26a6042383c827`; exact no-provider preflight passed. Actual Node24 is explicitly selected by launch PATH, not merely present in the sealed runtime. 213 native gold/admission proofs are a transparent subset of actual227 ARM/Node24 executions. Frozen ARM444f5047/x6457042f53, concurrency256 (213 cases), existing NVFP4 server unchanged. GPU96%; syncPID2213713/importexit0. Fresh clean authority binding replaced the completed56 assignment under the shared lock.

Root found a real TextWorld false-positive grading bug: external world services were accepted by matching a certificate string without consulting trusted host completion state. Eleven iterate reference cases caught IterationLimitError and returned literal gold without world actions. Entire world scope is held from the new DGX pool until strengthened completion grading is deployed (14 legitimate quest cases also held as a precaution). Preserve raw records/golds and old proofs; do not treat these false positives as clean positives or semantic DPO negatives. Source audit agent is implementing isolated collector regressions and auditing existing main-run exposure; rejection agent repairs bounded TextWorld reference iteration separately. Active frozen runtimes remain unchanged.

Local Luna600 remains active (104/300 and138/300 at04:54); Space Bunny256 reached217/256. Quality agent prepares exact residual Luna141 and rederived Bunny669 with explicit payload/history/protected-source filters for conditional successor launch. Do not claim the rederived669 is the exact old residual member list, and do not label component-sharing curricula as novel standalone sources. Next50-minute deadline starts once these active repairs and handoffs are complete.


### 05:01 UTC follow-up

World guard committed `fc17cc4`; root independently reran both isolated regressions (2/2 passed). Existing same-source training data is a genuine Luna solve, with observed non-null certificate after real actions: no automatic hold. Pinned exposure report `runs/training-source-exclusion-20261003/textworld-certificate-training-exposure/exposure-audit-v1.json` SHA `a514965626e90220b0ec7b8ece514487832ce01b92d26e01d02fba7c047b6fd3` records step1860/cursor14877 and36 valid same-source rows (6 consumed/30 future), next20824. Agent's prose retained an earlier1840 snapshot; use receipt state, not stale prose. Guard is future code, not deployed in active frozenv39; all new world generation remains held until sealed upgraded runtime. Cached TextWorld IR lacked the bounded iteration already present in current source; refreshed IR/provenance checks are necessary in addition to fixing reference gold fallback.

Root armed conditional v42 successor handoffs: Luna71/70 planSHA `3ec05d3c20c66ab74fa5a3fdd3cb10264d61df31ca01dc0298659f5111ff92a6`, controllerPID2217903; Bunny669 rolloverSHA `e9b0786d4f2e67720725f1dea416bb418401c42e96cbb16ee45f16c3f8b10e1f`, controllerPID2217904. Paths under `runs/generation-continuity-20261003/refill-workflow-v42`; waiting on current300/300 and256 exact predecessor accounting. Root corrected cloned stale review counts/proof paths in separate approved copies, preserving drafts. Root verified810 exact source JSON/native/queue joins with zero same-teacher history/active payload overlap and no worlds. Source-row SHA in old proof is sorted compact UTF8 JSON, not physical line bytes; correction sidecar records both and verifies exact810 original line contents. Counts are payload-level freshness within existing source coverage. Future packet construction should create typed schemas from final queue/proof rather than clone predecessor metadata.

DGX first15 final artifacts imported by05:00 (lastpoll4 new; cumulative15). LFM training1850+ continues. Next supply agent prepares source-safe nonworld DGX successor, without provider calls/authority changes; refreshed world reference repair is independently running. Both work streams remain held for root review.


### TextWorld reference follow-through — 05:03 UTC

Reference source now uses explicit finite iteration limits, surfaces failed playable progress, and returns the computed live certificate with eval finish:true instead of a literal expected value. Commit recorded by root separately. Versioned fixture `runs/textworld-iterate-reference-repair-20261003/candidate-v2/reference-audit.json`, SHA `e351f3c619130d1a28aa407319a4053ff6f0bfcfbe4ec2f2dc64b65576b63f37`: x64v39 actual native playable done/7approved decisions after three real actions; missing-object case blocked/quiesced with1helddecision, no positive target. This fixture uses oldv39 plus action evidence; combined strengthened-grader replay and refreshed fullworld/ARM proof remain required before world launch. Old cachedIR and gold preserved.


## Cadence sweep — 2026-10-03 05:53 UTC (ongoing)

50-minute sleep completed; main source-clean-v2 Muon training reached2130, later2200. Active run remains unchanged while a checkpoint-safe periodic evaluation upgrade is prepared at the user's explicit request. Current legacy evaluator only evaluates first100 heldout rows before/after, example-mean loss; it is not periodic and does not score all5410 protected rows. Future periodic metric will use fixed deterministic128 heldout rows, masked-token weighted loss, every500 optimizer steps plus an initial observation on resume. Keep baseline/final labels separate; preserve weights/Muon/scheduler/RNG/cursor, frozen code packaging, source exclusion marker and protected split. Source audit agent implements future-only code and CPU production-path proof; do not activate unreviewed drafts.

User also requested W&B. Valid `api.wandb.ai` credential from DGX netrc authenticated accountwerg. Private local configuration is `/home/werg/.config/natlang/wandb.env` mode0600 (never print/read it into tool output). Separate CPU-only observer service `natlang-main-training-wandb.service` is enabled and active; code commit3b3afe9, SDK0.30.0 in isolated `.venv-wandb-report`, pinned `training/requirements-observability.txt`. Training container has no SDK and was not changed. Run `https://wandb.ai/werg/natlang-training/runs/lfm25v13muon20261002`, saved metric backfill and live telemetry. Server API independently confirmed optimizer_step2170 and actual numeric history records (`runs/lfm25-350m-broad-20261002/wandb-main-reporting-v1/server-confirmation.json`). Online local SDK queuing and remote confirmation are distinguished. Reporter survives resumes with stable runID; emits only scalar metrics/public config, not source examples/model weights; bounded512MiB/0.5CPU. Periodic completion JSONL hookup awaits upgraded trainer path.

Space Bunny256 completed and669 successor launched automatically, >528 completions by06:00. Luna2 completed300 and launched70 successor; Luna1 still238+/300,71 successor waiting. DGX213 imported203 by05:52 and205 raw results by05:59; seven remaining cases had fresh durable activity, so no blanket stall/timeout. Index155 has RangeError Invalid string length with partial evidence, not a semantic DPO negative; rejection agent investigates capture/terminal accounting. Semantic failures include typed boolean strings and relation/count errors; retain exact native boolean contracts.

Root found a serious proposed DGX supply bug: builder treated all4803 quality-v4 records as primitives, including existing directory-v4 composites; it copied only root question/state and omitted nested component folder files. Native literal-gold reference/admission replay could pass despite absent visible evidence. Exactly398/587 r3 composites are affected,189 not affected by that specific omission (`pool-v7-r3/root-visible-input-exclusion.json`). Whole r3/r6 proposal held; no provider launch occurred. Outer directory reducer inputs:{} is legal; the failure is missing component dependencies, not that root shape. Quality agent rebuilds from genuine state-bearing primitives only with independent visible-input/source/gold dependency-closure checks before ARM native proof. New aggregation of already-seen source evidence is useful task-shape coverage, never label as novel source coverage. Preserve diagnostic proofs/drafts; their native pass is insufficient quality evidence.


## 2026-10-03 06:33 UTC — periodic training/W&B and source-complete supply

- Resume main training from exact step2281 checkpoint into sealed runtime-v11, with initial and500-step periodic heldout evaluation. Preserve full Muon/RNG/cursor/scheduler state and source-clean split; legacy example-mean baseline remains separately labelled. Separate CPU W&B observer confirms uploaded step2300 and periodic loss0.917649, no SDK/GPU overhead in trainer.
- DGX predecessor212/213 is explicitly reconciled with one held capture infrastructure failure, never falsely called complete. Launch reviewed409 source-complete directory composites with1844 exact visible input/gold joins and ARM native proof. Existing directory-shaped source records are excluded from primitive composition. Native admission alone does not establish visible source sufficiency. No new world tasks until stronger trusted-world grading deployed.
- Review joins use embedded IR identity and runtime canonical digest; selection physical-JSON SHA and top-level teacher trajectory ID have different meanings. Failed root assertions preserved and corrected before launch. See latest HANDOVER activation and per-packet receipts.

## 2026-10-03 06:49 UTC — bounded diagnostics and continued provider supply

- Commit bounded preview/capture while preserving live values and modest paging. Diagnostic projections cannot become training actions without same-turn exact raw model-call proof; projected-only actions are explicitly unlinked.95focused checks. Future sealed runtime proof required before repairs/world generation.
- Start reviewed Bunny409 successor after669 exact terminal predecessor accounting, no distillation flag; source groups are reused and cross-provider task comparison is explicit. Remove mutable status from immutable launch pins and create absent realworkerstatus only through launcher.
- Periodic loss class/source labels initially unknown; loss/counts valid, prepare hashbound support metadata correction. Complete-task CPU eval preflight caught oldoutputwatcher and packet-schema mismatch; fix before any model call.

## 2026-10-03 07:06 UTC — evaluation activation and Luna refill

- Start24-case CPU complete-task evaluator after actual no-model setup proof; historical2200baselinepairedfirst, future500stepcurrentv3snapshotscoalesced. Preserve gold values; two stale digest fields corrected in a separate packet/receipt. Resource failures never semantic DPO negatives.
- Start205+204 Luna sourcegroup-disjoint r7 queues after exact71+70predecessor terminal accounting. Actuallaunchschema controls/pins and sourceclosure labels corrected before claim. All sources reused; exactcompositepayloads newforLuna.
- Hold Bunnyv43case7 transient-looking providerfinisherror withrawpartial for classification/recovery; continueothercases. Future runtime boundedcapture/worldproof completed but staleprovenancemap caught beforeseal/deploy; require freshr2manifest.

## 2026-10-03 08:25 UTC — CPU scoring, cleanup and provenance review

- Classify outcomes from trusted status/checks, never infrastructure-keyword searches through model prose. Preserve baseline raw24 and attach offline hashbound reclassification;1/24success,13semantic,1contract,9incomplete. Reuse baseline rather than spending compute to repeat it. Newv4 adapter evaluator active with exact pinned plan and scoped Docker ownership cleanup; GPUtraining uninterrupted.
- Derived train-group membership alone cannot establish source-alias disjointness from protected heldout. v8selected components exceed r7 set, so require full corpus closure scan before DGX512 launch, even with native/reference/visible-input passes.
- Review receipts must derive fields from explicitly named source partials. Index155 publication held for second copied Bunny field; preserve erroneous receipts and correct versioned proof before any training inclusion.
- W&B audited support namespace uses pinned aggregate counts; raw examples/source IDs remain local and legacy unknown fields remain intact.


## 2026-10-03 — DGX development pipeline replication

- Replicate the reusable pipeline and lineage, not only admitted training outputs. Copying candidate/rejected material does not grant admission; protected splits/source exclusions remain enforced.
- Keep Git parents real and link only untracked data branches to external storage. Preserve pre-existing development conflicts and report them. Corrected the initial parent-symlink design without discarding data.
- Home owns mirrored corpus; DGX experiments use a unique return namespace pulled first. Preserve replaced revisions and avoid deletes/in-place updates. Source code stays under Git review rather than periodic overwrites.
- Bootstrap the current recipe closure with pinned hashes before a serialized wider mirror; persist retry/reboot recovery and distinguish closure proof from full mirror completion.
- Full heldout closure uses every recursive source/program/group alias; do not force alias counts to equal observed group counts. DGX512 v8 launched only after that proof.


## 2026-10-03 — exact offline recovery registration

Register index155 only after independent full request/context/tool hashes, retained raw calls, typed gold and protected source checks pass. The trace is explicitly a retained offline interpreter trace, not a recovered original trace (absent from the raw partial). Preserve the original capture infrastructure failure; it is not a semantic DPO negative. Locale-aware runtime canonicalization differs from Python lexical key order for trace maps; use the runtime algorithm for the independent trace check. A stale registration-helper review field failed closed; corrected v3 checked actual schema/counts before atomic publication. Current training inputs remain fixed; future snapshots discover the new exact row.


## Development sync retry repair

Repeated priority-copy exit23 matched read-only rsync partial resume failure reported at https://lists.samba.org/archive/rsync/2026-April/033295.html. Existing private staged results/turns had mode0444. Stop the sole retry writer cleanly, preserve staged bytes, then make only exclusively owned single-link staging files owner-writable before rsync; retain final/source modes and initial recipe hash gate. Fail closed on staged hardlinks/shared ownership. Apply the same preparation to recurring push and owned pull staging.


## Historical recipe code materialization for development replication

Priority data copy completed after the read-only staging fix. Verification then correctly rejected one mutable build path: v33 recipe pinned `ts-host/dist/teacher/source-review.js` SHA d0c15955..., while the current dev build is newer. All80 other logical input hashes match. Preserve exact old code bytes from sealed runtime-v34-r4 in `runs/dgx-development-sync-20261003/recipe-input-closure/...`; record an explicit original-path mapping in immutable recipe-replication-input-manifest-v3.json SHA c60223e630eadcef771956d8cbc5bd130040c8b4db8b172fa0620d585ecdbf95. All81 original content hashes independently match. Do not alter the original recipe or overwrite current code to force an old provenance check to pass. Broad mirror waits for this mapped exact closure proof.


## 2026-10-03 10:06 — reviewed v9 launch and proof schema clarity

- Finish and reconcile v8 all512 before fresh v9-v2 launch. Use independently checked visibility, full protected alias union, actual ARM native admission, exact history rescan, no-provider preflight and model/slot check. Runtimev41 capture/provider retry fixes only apply to the fresh v9 campaign; old provenance untouched.
- Separate case/row counts from decision counts: materializer_accepted is cases1024; materialized_decisions4410. Preserve erroneous drafts with explicit correction proofs. New approval helper materializes fresh artifacts and refuses existing files.
- Physical IR serialization digest is different from runtime canonical recordDigest; never equate them. Exact IR structural joins plus actual runtime identity proof establish the connection. Python lexical sort and JS localeCompare can differ.
- These1024 composites reuse known scenarios and add task composition/cross-teacher variety, not new source coverage. Keep family-concentration audit when building the next recipe.

## 2026-10-03 — retain the full development pipeline

User clarified that replication includes the non-obsolete pipeline and its
source/intermediate evidence. Preserve historical records required for lineage;
copying rejected/candidate data does not admit it to training. Carry committed
code through Git and preserve uncommitted adapters/prototypes/controller work
as an explicit hashed source snapshot, without overwriting DGX development
edits or silently applying unreviewed code. Initial broad mirror completion and
81-input priority verification are distinct states.

## 2026-10-03 — conditional continuation and truthful start ambiguity

Arm the reviewed DGX1024 successor only after exact predecessor imports and all
source/native/model/resource/authority gates. A timeout in a post-start status
probe is unknown launch state, not evidence that no launch happened. Preserve
the claim and require reconciliation before retry. Independent review found
and root fixed this exception path; old unarmed configs remain retained.

CPU evaluation resource deadlines must be classified from the runner-owned
abort signal rather than exception text or model-authored prose. Preserve the
original reports and append a trusted audit correction. Cases held before any
provider request must be distinguished from student failures. Future code fix
is in progress; active sealed evaluation runtime remains unchanged.

## 2026-10-03 — reconcile failed parents without losing task supply

Bunny queues with an incomplete terminal attempt can advance only after root
review pins each exact failed journal event, verifies every other queued key,
and preserves failed raw evidence. Keep their completion disposition
finished_with_reviewed_failures, distinct from successful exports. A provider
error partial is neither a training positive nor a semantic DPO negative.
No active sealed teacher pins changed. Fresh reviewed successor512 started
with the current bounded-capture/retry runtime; exact512 parent tasks are
intentional cross-teacher comparisons, not new source coverage.

## 2026-10-03 — own subprocess lifetime in systemd services

A handoff process must remain alive until workers/importers it launched exit:
start_new_session does not escape a systemd cgroup. The new Bunny transient
rollover exposed this bug; root recovered its untouched512 queue once under a
dedicated worker service after proving no job evidence and oldPID absent.
Luna/DGX waiting controllers were replaced before any successor launch, with
helpers that wait outside authority locks for their children. Remove small
controller-only memory caps when the cgroup now owns workers; retain existing
per-case generation resource policy. Keep failed launch/recovery records and
truthful authority reconciliation, never claim a dead PID is ongoing generation.

## 2026-10-03 — portable dev setup and truthful review metadata

Normal Node builds stage the exported workspace package; full setup builds
applications. Honor the selected dataset cache consistently. Tests isolate
model health and user config rather than changing production timeout policy.
Source-review fixtures must use the exact prompt/revision/gold evidence held
by policy; do not restore broad ID-only holds to satisfy stale fixtures.
Preserve independent DGX development commits when syncing source.

Future native packets must directly report admitted case count separately
from materialized decision count. Only historical v9-v2/v3 retain explicit
count-correction provenance; new correct packets need no manufactured error.
Missing platform metadata is appended as hash-bound versioned attestation and
new approvals, never overwritten in existing approved artifacts. Respect
resource gates when memory is briefly insufficient and retry after recovery.

## 2026-10-03 — cheap student rewriting before sampling complexity

User proposes adapting Finetuning with Sampling after substantial direct SFT,
then explicitly prefers the simplest quick gain. Start with student-conditioned
teacher rewriting and strict whole-episode runtime/native admission, ordinary
SFT under original contexts, one attempt plus a retry only after failure.
No MCMC/proposal correction or likelihood scorer in phase1. Do not describe
this as a reproduction of theoretical projection sampling. Privileged prompts
live only in raw proposal sidecars; runtime observations are fresh and stored
SFT contexts are original. Preserve every unsuccessful attempt separately.

Existing full Muon main run remains active. New lane is implemented/preflighted,
not launched or added to live training. First64-case draft is source-safe but
uses an early snapshot placeholder; choose a late-SFT checkpoint for actual
collection. After rendering/admission, resume a copied COMPLETE Muon checkpoint
through a reviewed data transition. The current main exclusion lineage and
future append must be reconciled; never silently reset optimizer/scheduler/RNG.
Details and remaining work: plans/STUDENT_REWRITE.md.

New student-server launches default to127.0.0.1 and expose an immutable loaded
adapter/revision identity for this collector. Container callers need host
networking or an explicit container bind with a loopback-published host port;
active sealed eval servers stay on their original implementation. Resume refuses
orphan proposal evidence instead of overwriting it. Bind an explicit64-program
subset separately from the full source-reviewed1024 IR pool.


## 2026-10-03 — replace rewrite-only objective with projection sampling

User clarified that the intended method is the paper's iterative expert-guided
search and ordinary-prompt student likelihood selection, followed by SFT. The
root's rewrite-only-first interpretation is superseded. Keep existing unlaunched
collector as reusable infrastructure; do not launch its baseline as fulfillment.
Document theoretical proposal-corrected MH versus released greedy likelihood
search explicitly. Online operation is under discussion; freeze each chain's
weights, replay changed tool actions, and preserve training/checkpoint lineage.
No live generation or training configuration changed by this decision.


## 2026-10-03 evening — monitoring and evaluator bind regression

Main full Muon training continues (6120/13165 at check; heldout step6000 loss
0.836124, zero skips). Qwenv9-v4 completed1024 with997admitted/27wrong_return;
Bunnyv45 finished512, Luna live. Prepare task refills before next sleep.
V4 execution evaluator step5500 mounted mutable working server source, despite
startup hash pins. Root's825756e default-loopback change therefore affected a
later Docker spawn: container bridge publication cannot reach container-loopback
binding. No model requests were made in that failed evaluation. Fix in new v6
with frozen server/helpers, explicit container0.0.0.0 bind, host127.0.0.1 publish
and per-spawn pin checks. Preserve v4 failure; do not label it student failure.
Earlier statement that sealed evaluator launches were unaffected was incorrect.

Stopped exact retired importer PID1334920, still repeatedly checksum-pulling a
paused old Oct2 campaign with1023/1024 imported after infrastructure failure.
Its data/partial failure remain retained; no full completion or DPO claim. Current
live authority is newer v9-v4. Receipt monitor-cadence-20261003/retired-importer-stop-v1.json.
New v5 preparation must use matching sort seed on both comparator operands;
v4's mixed v9v4/v9v3 operands were inconsistent. Preserve historical v4 selection.


## 2026-10-03 — preserve incoming Neuralese build outputs on rerun

After reviewing/merging origin/main, remove destructive finalized-output deletion
from both Neuralese build wrappers. Require fresh, path-safe tags and atomically
reserve run directories; default tags include UTC time. Store each build's protected
source snapshot in its own run. Existing partial builds must be resumed with explicit
CLI stages, not overwritten by rerunning the whole wrapper. No conversions, tests,
builds, model runs or new training were launched as part of Git integration.

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

### 2026-10-04 current full-candidate audit isolation

Pinned schema-audit dependencies in a separate DGX venv; no training dependency
changes. Production candidate validation streams45files under8GiB service cap.
Optional sequential/NOREUSE read advice reduces competition with serving cache
without global cache flushes. Candidate held until current audit and source/split
review complete. Exact manifest/commit/dependencies are bound in input receipt;
no admission claim based only on the earlier converter's preloaded checker.

### 2026-10-04 malformed Bunny reply observation

Job436 exported40matchingrequeststart/endpairs,28emptyvisible/no-actionreplies,
six marker/NUL replies,2incomplete typedreturntoolcalls. No streaminvalid/retry
observed; no rawHTTP/SSEbodyretained, so blamecannotbeconclusive. Preservecase
as incomplete,notpositiveorsemanticwrong-answernegative. Future Pi source now
retains anomaly-only parsed-SDK-content SHA/escaped256byteexcerpt,visibletext
SHA/bytes,reasoningpresence/bytes,stopreason/usage and capped32blocktypes.
Collector savesboundeddiagnostic,notrawrequestheaders/credentials. Source label
explicitly parsedSDKcontent,notHTTPbody. No stripping,coercion,newtimeoutor
stoppingpolicy; activefrozenv41unchanged. Newfreezewillneedusualproofbeforeuse.
No tests/buildrun; source reviewed anddiffcheckonly.

### 2026-10-04 variable-size reviewed Qwen pools

The next residual pool has exactly 319 cases. Approval now derives a positive
integer count from the reviewed packet, then independently checks that count
against selected rows, assignment, native proof and source checks. Padding to
1,024 would add unnecessary repeated compositions. The exact v6 assignment is
screened regardless of changing import dispositions. Native ARM replay approved
all 319 cases and 1,369 decisions. This remains a customer-service utilization
buffer; broader source supply is being inventoried separately.

Original v7 drafts were preserved. A canonical join receipt and packet-v2 bind
the existing proof without rerunning model generation. Assignment-v2 corrects
stale inherited closure/draft hash metadata; the materializer now derives these
fields from verified evidence for future pools. Root preflight passed with zero
jobs started. Controller config-v2 is pinned and waits for terminal v6 status,
full imports and released capacity. An initial config omitted predecessor host
and authority key; its failed readiness receipt is preserved, with no launch.

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
