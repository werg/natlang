# DGX current state

Generated 2026-10-16 11:46 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-16T11:46:18.461459Z by dgx-codex-01a13659.

DGX 2026-10-16 11:46UTC. Continuous supervision ACTIVE. Owner: integrate all deferred improvements; entire evolving recipe on both students, two training sandboxes plus one stronger self-hosted Qwen generation sandbox.

Training: LFM drained exact6711 parent SHA e764d30a... (2366517423bytes,27287creditedvisits) then standardcontinuation plan passed18.53s; recipe655642 restoring992297train (+75700),25955held (+156),all2449pending retained. Earlier6654 hostOOM lost34unsavedupdates, restored6620; secondOOM killedadopter notusefultrainer. Cgroupfree31.6GiB overstatedactualVMMemAvailable11.44GiB; allfurtherLFpreparation serialized. A pendingpointer was provenalreadyretired, notnewadmission. Sharedretirefixb14e91e8e removesonlyexactretiredrequest underlock,preservesnewer,rejectscorrupt;26tests+quickPASS. Mellumcontinuescurrent4bda runtime,7759lastfreshstep;26907newtrain+169held adopted, actualvisitsseparate. NativeinterfaceFP32AdamW.0003both;LFreasoning.0003,Mreasoning/backbone3e-6. Original26312communicationfoundation incompleteBOTH; noVAEreadiness/qualitylift claim.

Efficiency: actualLFH1006620 prefixqualificationPASS candidate8fc:13431to8188physical(-39.04%),bitwisehidden/logits/loss,inputgradrelL2 4.14e-6. PrefixOFFpendingseparateadoption. Exactsource-indexcontinuation399162917+182a9ecfd published,independentreview,16focused+quickPASS,actual3042to8fcpolicyproofPASS; avoidsunchangedsourceparse withoutadmission/exposure. Filecheckpointstoragea7f/729 strideexact/holderlease validated,43actualnativeCUDAderivativePASS. Frozen3042-basedcandidate248cd0924 has6changedruntimefiles,34focusedPASS7.10s,independentreviewclear; actualcloudlonggraphcapacity/throughputstillpending, no truncation/exposureclaim.

Data: D24 full147985SFT+12413preferencepairs normalized,ZIPcontentmultisetproof62e918. FIRST6187sessions/model completeverifiedproposalSHA8ae110f91cd1ac7584660c2521553a977b10a3e8070201f15d42ce00f60ad009;LF65.25Mtokens/max54642,M70.90M/max79772. Cloudownerrequestedregister/stage/adoptthismemberNOW,notwaitall15exports. No actualD24admission/exposureyet. Latermemberresumeexactpathbindingfixprepared; MhostexportpeaktemporarilyblocksnewCPUwork. Six-refjob577921 hadfailedatD14wrongVMroot; sourcesactuallyLF:D13/D32,M:D14/D19/D36/D37. Splitretryguarded;Mrefsjobexited75guard,preservingoutputs. D20boundedretryalsohit4GiBRSS; rootlocalizedauxiliaryParquetdecodebeforerejection. Metadata-onlyexactrowaccounting52e991264 passes46tests;exportercovered-ID/set(range) retentionalsofixingbeforefreshretry. Searchr7/rollingadmissionactive,shard0014registered. Protectedheld/fulltrajectoriesretained. HFmanualD09still403,automatic3hcheckthroughOct19,next12:45UTC.

Teacher: oldRadixArkengineactualfedagain11:22:42 afterNVIDIAstartupfailure;564oldbatch14finished preserved+460casecontinuationwithownrollbackactor/f01runtime. NVIDIAexactfailureio_uringregisterbuffersENOMEM,memlock8MiB,nohostOOM. ShippedInstantTensorMMAPreaderCPUpreflightPASS; secondactualpinnedCSFvLLMstartup694949 now74.48GBGPU/hostAnon8.7GB,VMavailable29.9GiB, noio_uringfailureyet;servingnotqualified. SamegenerationVM,nofourthallocation. V18oldbatch14verified-successdeltaexport2529active. Sharedconverterb5c61b973 nowacceptsexacttransferfile.collection_source boundverifiedcollection+selection/protectionpacket,legacylayoutunchanged;36tests+quickPASS. Newcontinuationrawrowsneedthatbindingbeforeadmission, no source relabel.

Behavior: fixedDGXdispatcherandMingesterhadbeenINACTIVE witholdLF5500/M5000activepointers. Explicitlysuperseded/droppedthoseunscoredrequests;bothcorrectedservicesnowACTIVE,f01exact699files43e248 verified. LF6620verified2.36GBexport fetched+sealed11:35:46,actualtaskeval-6ca5c1c6-0006620running.14results:1accepted/13answerrejections;case3repeatedactions hitcollectioncap,case11stillrunning/8192tokenreply. No completed16-taskscoreyet. Analyzeerrors/materializationbeforeattributingqualityorchangingobjective. Mnextcurrentcheckpointweights-onlyexportscheduledusingexisting96abpruner,actualsavedstep/parenthashdeterminedfromimmutablepin,notmislabelled7500. CorrectedMingesterwaitsNEWexport;evalout /mnt/external/natlang-development-data/native-mellum-corrected-eval-20261016 avoids24GBNVMeexhaustion. ThosehistoricalchecksdoNOTqualifycurrentcloudweights.

Nativelearning: e4a7ecc00 internalfiniteDAG/optimizedstate/semanticarchive/typedHTTPcomponentreviewing,NOTadoptedyet. Rootfoundnumericverifieronlycomparedstate==updateartifact;agentaddingactualsharedoptimizerrecompute/jointforgerytests andexplicitqualifiedarithmeticcontract(nofakeCPU/CUDA/crossCPUexactness). Receipt-onlymixedarchivefollowupPASSbothports. RLbundledependency/topologicalrestoreandnativeexternalderivationvalidationgapsalsoagent-owned. Publicnatlang.grad captured-turnloweringandorder2remainOPEN/guarded. NofullsizecloudRL/liftclaim.

Checks/storage: full72957ded91 GREENall11steps. Stale399fullcheckstoppedbecauseknownhelperratchetfixed182;current99da4b04a fullcheckrunning. Mainlatestb5c61b973 own26+36focused/quickPASS. RemovedsixdeadOct13-15pytestfixtures20500574208bytes(no liveopen/maps/cwd/repopins),anddiscardedoldprojectedMellumv2servingexport23423608631bytes plusstalesymlink/manifest/archiveindexrecord;activeparents/reusabledataunchanged. Root29GiB/external71GiBfree. No whole-recipe/completedarchitecture/optimalutilizationclaim.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `corrected-mellum-eval-ingest-root-20261016` | medium | 3.0 / 0.3 |  |
| `corrected-native-task-dispatch-root-20261016` | medium | 3.0 / 0.3 |  |
| `cw-lfm-foundation-ferry-2264` | low | 1.0 / 0.1 |  |
| `cw-mellum-current-inference-export-2526` | medium | 1.0 / 0.1 |  |
| `natlang-full-check-99da4b04ac` | low | 13.5 / 4.3 |  |
| `natlang-memory-guard` |  |  |  |
| `search-completed-selection-dedup2-20261016` | low | 2.0 / 0.0 |  |
| `search-completed-shards-admission-current-root-20261016` | low | 8.0 / 0.7 |  |
| `searchowl-native-render-optimized-resume-20261016-r7` | low | 16.0 / 4.5 |  |
| `taskeval-6ca5c1c6-0006620` | medium | 66.7 / 10.7 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| native_state_sft | lfm | `native-state-production-lfm-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the lfm backbone. |
| native_state_sft | mellum | `native-state-production-mellum-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the mellum backbone. |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (77): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `native-state-production-base-20261015-v1`, `native-state-production-lfm-20261015-v1`, `native-state-production-lfm-rest-qwen-v3-append-20261015-v1`, `native-state-production-lfm-teacher-comm-append-20261015-v1`, `native-state-production-mellum-20261015-v1`, `native-state-production-mellum-rest-qwen-v3-append-20261015-v1`, `native-state-production-mellum-teacher-comm-append-20261015-v1`, `native-state-semantic-curriculum-rest27-foundation-20261015-v1`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1763 live entries in `training/neuralese_corpora.json` (317 dgx, 1446 pop); 406 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-16T10:00 dgx-codex-01a13659: Deploy adaptive source and restore improvements with exact current-parent binding
- 2026-10-16T10:20 dgx-codex-01a13659: Shared source reader and numerical context 74e47f187
- 2026-10-16T10:25 dgx-codex-01a13659: Combined active and pending recipe handoff 60ecb1505
- 2026-10-16T10:32 dgx-codex-01a13659: Explicit higher-order reference arithmetic d2f9d1baa
- 2026-10-16T11:18 dgx-codex-01a13659: Exact prefix source-index continuation 399162917
- 2026-10-16T11:24 dgx-codex-01a13659: Deferred integration and completed-fixture cleanup 99da4b04a
