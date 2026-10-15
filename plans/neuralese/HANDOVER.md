# DGX current state

Generated 2026-10-15 14:17 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T14:16:53.949232Z by dgx-supervisor.

2026-10-15 14:16 UTC — DGX owns all operations; Pop CLOSED. Exactly TWO student training sandboxes plus ONE generation sandbox; no further allocation planned. Mellum/LFM are STUDENTS. Owner corrected weak Mellum teacher choice: deploy stronger self-hosted Qwen3.8-Flash-Next NVFP4 on existing dadd0462-0175-470d-b1ee-4d52edfe110e (RTXPRO6000 Blackwell96GB,8CPU64GiB). No Luna/API generation. RadixArk source revision7b719225242aacd3dbd3f9407468c2ee9a9d2594, manifest57711090fba54251872024c6b6f59b99675fe678568a6bb9067fe405814d1cba;135253622894B/419files,14:07~72% downloaded. PennyRoyal3.0.1 sourcefa9c2673131621cafb9a527e61d97c25eac130a9; isolated nativebuild+Rust1.97.1+bounded file-backed PLE reader underway. /work actualvirtiofs, NOT verified physicalNVMe; measured O_DIRECT~11k4kIOPS/343MBps sequential suitable to attempt.4GB actualpinnedallocation PASS despite8MiBmemlock. Strongteacher notservingyet; no positive claim. Retire weak Mellum actor3871 atcaseboundarywhenstrongstaged; weakH100actor retired13:49. Expand verified train-only teacher queue beyond32initialevidencetasks acrossnatlangdomains. Four prior correct DGXbase samples stillassignedadmission. Other sandbox456d8127 belongsactivehindsight-kb, untouched.
Published native semantic curriculum62d4bd7a7 and recipe foundationbac408105: shared VAE→jointproducer/control/consumer→behavioraldistill, completeperrow/modelpairexposure, deferred MuonSR, exact allmodel/VAE/optimizer/RNGresume+dataappend/laterjoin; actualfinalruntime+heldTaskEvalrequired. Actual FULL-SIZE joint update PASS14:03: QADFP32+MellumBF16 full6989/7070toksourcegraphs, triton-experts+checkpointing, finitegrad and actualbothbackbone+adapterupdates,197.75s,77394900992BpeakCUDA(~72.1GiB),6288605184BRSS. Report native-real-joint-update-20261015.json; not production/tasklift/fullrestore qualification. 96GBsandbox preferredsharedplacement. Namespace agreed /work/native-production: runtime alias->immutablecommit;models/{lfm25-qad,mellum21};data/<immutable-corpus-id>/<tokenizer>. BothSFTout /work/native-production/{lfm25,mellum21}/native_full_sft, readiness native-training-receipt.json+siblingcheckpoint/report/runtime/task/score. Peer mirrors preserve SAMEabsolutepaths; publishreadinessafterallboundbytespresent+SHAverified. Globalout /work/native-production/shared/native_shared_stage. Controller-owned dynamicparticipants, notdummy staticmanifest.
Native inference rawKV/preRoPE+finiteconv qualified actualQAD/Mellum/H100 compiler. New compactsemanticwire d26a5dada/887252833: text+actualsourceIDs/embeddings/positions/offsets+per-tokencode, exactadapterSHA underexistingnzs1 durable lifecycle, no rawlayerKV. Even same-modelcompactreadsdecodeVAEtargetdeltas; unknownadapter explicittextfallback; RLarchivesactualdecodedtargetobservation+additionalrawproducerforreplay. ActualsampledBPE/UTF8tokensretained via explicit sourceoffsetpolicy, no substitution ofcanonicalIDs. Python32integrated+13offset/serve/RL PASS, TS12codecs/archive/runtime PASS+layering/build/quick. Model-facingmeasurementdebt0/48recorded for nextnative teacherwindow. Independentreviewactive: realHFnoncanonicalUTF8, compactRL, view/chunk/combine andreceipt-onlyartifactarchival beingclosed. DoNOTclaimcomplete dialect/cutoveryet.
Advancednativelearning integratedthrougha95ed7611 as e8f5718c5: cachedscore/decide, firstordergraphCE+registeredpolicyobjective grad, namednnt1 typedartifacts Python/TS/durablearchives, immutablecomposite/interfaceSGD/Adamupdates+optimizerstatecontinuation. Sharedparticipantloaderpreserved. Fullbackbone learnerupdatepersistence/adapters/directgrad/order2/guidance stillOPEN; nofalsecapabilities. Localintegration32PASS+quick. Standingfullc048 run foundteacher->neuralese layeringcycle, fixed887 bymovingnative-state-client implementationtomodellayer, layeringactualPASS. Standingfullservice shouldnextcoverlatestmain; do notlaunchduplicatebroadfullsuite.
BotholdcloudstudentrunsremainDIAGNOSTICpendingnativecutover. Last13:52 LFM30876 finite.4714,GPU44%sample;Mellum4338 finite.0671,GPU97%. FullprivateLFM30360 SHA87a261bb... and Mellum4200 SHA3f1594e2... independentlyverified; exacthashesprivateperiodicstate. Bothperiodicexportersactive. Requested14:15freshallprocesscheckfromcloudagent includingDolci/Big2. Dolci135/644lastcountadvancing;300originalwrong_returns assignedhistory/indexruntimecause,3unchangedIR/oraclefreshreplaysPASS. No unverified dataadmission.
Dataagent26dad20fd integratedlocalmaina1f2dbef5: FULLnativegraphs56,378each=53,223train3155held, independentfullscan0sourcegroupintersections, immutablemanifestregistered, exactQADmodelidentity/tokenizertwins, sharedbase+2nativeSFTrecipeinheritors, globalcontrollerhandler+inheritedrecipeidentityregression66PASS, conversion8PASS+quick. REST27 full14.5GB/147414records/55484edges converter was2.5%in20min CPU-bound; dataagentnowprofiles/optimizesbatch/cache/resume preservingexactrows/splits/sourceSHAs thenrestartswhenfastervalidated. Correctboundedpairproof/tmp/native-pair-proof-20261015z hasactualsourceedgeSHA26053ffead5b96d640fce4dcbd50a1cdaee781bf8702dbee0b3d35bffe6c7226; olderproof/tINVALIDmalformeddigest, doNOTuse. Boundedproofneverproductioncurriculum. EntireREST27train+heldpairsneedimmutablehashesbeforeglobalsharedconfigbinds. FULLfirstSFTphase notclaimwholedataexposure. R11/REST27/SearchOWL admittedsourcecoverage, overcap7002pagingandPop224roots955decisions and remainingthirdpartycoverage/conversions explicitlyassigned, notsilentlyomitted.
Storage14:01NVMe66GiB/external68GiBfree. Keepactiveparents/reusabledata/protectedheld. No47GBlocaldisposablecheckpoints. Owner untrackedNatlang_Neuralese_Training_Data_Expansion_Plan.md neverstage. AllDGXtests/probes/buildsledgered. Continueactualcheckin<=50min; no sleepwhileactionableworkremains.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `native-recipes-integrated-20261015` | medium | 3.0 / 0.7 |  |
| `native-rest27-fingerprint-optimized-proof-20261015` | low | 3.0 / 0.6 |  |
| `native-rest27-graph-build-20261015y` | low | 6.0 / 0.8 |  |
| `natlang-full-check-c04803f360` | low | 13.5 / 0.6 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p22-73710` | low | 8.0 / 2.1 |  |

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
