# DGX current state

Generated 2026-10-15 14:53 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
This page holds the present only. History is in git (`git log -p -- plans/neuralese/HANDOVER.md`), in the
decision records and in the coordination log. Live processes and the status pages are the authority.

## Read first

- `AGENTS.md`: owner rules (start here)
- `plans/REVIEW_20261012.md`: strategic review and the accepted direction
- `plans/neuralese/SHARED_STATE_MIGRATION_20261015.md`: native-state replacement and cutover requirements
- `plans/neuralese/NEURALESE_DIRECT_V1.md`: legacy diagnostic training design
- `plans/neuralese/LINE_PARITY.md`: Mellum/LFM line parity
- `plans/neuralese/DECISIONS.md`: recorded decisions
- `plans/MACHINE_COORDINATION.md`: coordination protocol, data sync, storage
- `plans/OWNER_REVIEW.md`: owner review queue
- `plans/MODEL_FACING_CHANGES.md`: model-facing changes and measurement debt
- `plans/NATLANG_PILLAR.md`: weekly NatLang-pillar slot

## DGX status page (this machine)

Updated 2026-10-15T14:53:30.618230Z by dgx-supervisor.

2026-10-15 14:54 UTC — DGX owns all operations; Pop CLOSED. Exactly two student training sandboxes plus one dedicated generation sandbox. Mellum/LFM are students; stronger self-hosted teacher only, no Luna/API generation, no new allocation.
CURRENT UPDATE 14:54 UTC (supersedes older snapshot below): Main4660fb252 integrates native RL full-state shared trainer/parallel rounds/joint adapter refresh (a9->1a2031031), immutable interface optimizer updates and hourly checkpoint policy, plus exact static multi-read edge repair (1ab->4660fb252). Fresh native SFT launched BOTH sandboxes14:40 from correct pinned bases: LFM recipe249444/child249508 FP32, Mellum recipe97069/child97086 BF16/triton-experts FP8. Last14:52 LFM source validation83%/51.21MB GPU1.98GB; Mellum trainfile fullyread47.77MB/base27.95GB resident; no optimizer update confirmed yet/noerrors. FULL53,223train/3,155held permodel,8192batchtokens/taskcadence5000; this is first stage, REST/thirdparty continuations still required. Runtime79e2de231 frozen oncloud. Native periodic private-checkpoint exporters armed; preserve old final optimizer until independently verified then retire old exporters. Old LFMfinal33880 SHA cf2f3b28aa8b6c43d32f51cba5d834bcae9ac63fa949c077cdfc155b474b53e4. Mellum old final/export verification ownedcloudagent.
StrongerQwen RadixArk ALL135253622894B/419files SHAverified. PennyRoyal3.0.1 build reachedlateFlashInfer compilerfault: GCC13 lacks __has_feature usedbyASANguard. Narrow non-ASAN compiler macro actualpreprocessPASS, retry14:50PID41673 with persistentcache; sourcepins unchanged. NoQwenserving/sampleclaim. WeakMellumactor retired14:22. Broader88-case protected trainqueue staged(56admittedprograms15sources+32evidence).
REST27 optimized8spawnworkers/resumable16trajectorychunks running:14:47 4,784/11,713trajectories,46,110graphs/model,11,714pairs/4,214pairomissions/1,340ordinaryrenderomissions permodel. New static-shared-producer-multi-read transformation recovers every exact matching typedreadpointer without claiming uniquehistoricaloccurrence. Actual25trajectoryfixture51->56pairs, zeroomissions; pair-onlyrepair5/5 previousomittededges. Finalimmutableoutputs admission and supplement pending; ordinaryrenderomissions assigned source-ID retrospectiveaudit/repair; newconversioncode records IDs/reasons. Big2finished12:20:1,333,791records70,656chains1,373,067edges(train1,328,824held44,243), zeroauth-unresolvedchaindrops; cloudarchive/independentexport241090 pending, nativeconversionassigned. Dolci146/644shards last14:15advancing. No data left silently omitted.
Independent compactsemantic/fullserve/RL review85f integratedd8a06f37f. ActualHFnoncanonicalBPE/UTF8+compactvalue/views/archive/RL and headlessactorCLI testsPASS. NativeRL a9 reports97focused/8production/2changedparticipantrefreshPASS, actualtinyLFMsharedparent->RFT->checkpoint->noopresume and bothportsRFT/GRPOoptimupdates; fullsizeRL/tasklift NOT established. Rootcurrentintegratedfocusedverificationrunning, quickPASS22ratchets/TS. Automatic shared-stage->RL bootstrap and actualproviderbindings activelyassignedRLagent; shared production controller/curriculumJSON pendingfullRESTpairhashes assigneddataagent, nofakeplaceholdercurriculum. Standingfull887 RED only TinyGraphModel fixture kwargs regression, fixedb503 and9targetedPASS; full897f113134 active, do not duplicate. Advancednativelearning fullbackbone serviceupdatepersistence/adapters/directgrad/order2/guidance remainOPEN.
Storage14:51 DGX54GiB NVMe/68GiBexternal free. Avoid local47GBdisposablecheckpoint. AllDGXtests/buildsledgered. Nextnormalcoordcheck <=15:44UTC; continuousactivecloudmonitoring.

Foundation: actual FULL-SIZE joint update PASS (QAD FP32+Mellum BF16, full6989/7070tokgraphs, sharedtriton-experts/checkpointing): bothbackbone+adapterupdates finite,197.75s,77,394,900,992B peakCUDA/6,288,605,184B RSS; /home/werg/data/dgxops/native-real-joint-update-20261015.json. This is execution/placement proof, not tasklift/fullcheckpointrestore. Sharedstage placement96GBpreferred. Namespace /work/native-production/{runtime,models,data}; SFT outputs {lfm25,mellum21}/native_full_sft, readiness native-training-receipt.json published afterallcheckpoint/report/runtime/task bytes mirrored SHAverified atsameabsolutepeerpaths. Shared out shared/native_shared_stage; no globalthirdmodelbarrier. Fullpaireddata phases VAE/joint-control/behavioraldistill preserve exactallmodel/VAE/optim/RNG states and adaptive data/hyperparameters.
Compact wire persists text+actualsourceIDs/embeddings/positions/offsets+per-tokensemanticcode, exactadapterSHA underexistingnzs1 archive; no rawKV storedin compactpayload. Knownadapter decodes targetstatedeltas; compactRL archivesactualdecodedobservations plusrawproducerforgradientreplay. Sourceoffsets retainactualnoncanonicalBPE/UTF8IDs explicitlybest-effort. Model-facingmeasurementdebt0/48 remains assignednextteacherwindow. ActualQwen downloadpin RadixArk7b719225242aacd3dbd3f9407468c2ee9a9d2594 manifest57711090fba54251872024c6b6f59b99675fe678568a6bb9067fe405814d1cba; enginefa9c2673131621cafb9a527e61d97c25eac130a9. /work virtiofs measured11k4kIOPS/343MBps, notverifiedphysicalNVMe; boundedPLE47.68GiB filebacked and4GiBpinnedallocation tested.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `cw-native-lfm-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-native-mellum-periodic-export` | low | 1.5 / 0.1 |  |
| `native-rest27-native-graph-build-20261015` | low | 16.0 / 6.2 |  |
| `natlang-full-check-897f113134` | low | 13.5 / 10.4 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p23-75072` | low | 8.0 / 2.3 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| native_state_sft | lfm | `native-state-production-lfm-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the lfm backbone. |
| native_state_sft | mellum | `native-state-production-mellum-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the mellum backbone. |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (72): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `native-state-production-base-20261015-v1`, `native-state-production-lfm-20261015-v1`, `native-state-production-mellum-20261015-v1`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1700 live entries in `training/neuralese_corpora.json` (254 dgx, 1446 pop); 344 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T11:55 dgx-supervisor: Native graph objective, VAE core and native serving integrated
- 2026-10-15T12:00 dgx-coreweave-sandbox: Explicit partial-runtime diagnostic continuation retains ineligibility
- 2026-10-15T12:33 dgx-supervisor: DGX generation closed; exact restart procedure published
- 2026-10-15T12:56 dgx-supervisor: Pop closed; continue generation only with own sandbox models
- 2026-10-15T13:41 dgx-supervisor: Shared semantic stage implemented; Pop closed and dedicated own-model generation allocated
- 2026-10-15T14:17 dgx-supervisor: Self-hosted stronger teacher and shared native execution
