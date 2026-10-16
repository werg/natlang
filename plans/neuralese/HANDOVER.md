# DGX current state

Generated 2026-10-16 11:23 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-16T11:23:55.053192Z by dgx-codex-01a13659.

DGX 2026-10-16 11:24UTC, continuous supervision ACTIVE. Owner: integrate ALL deferred improvements. Exactly two training sandboxes and one stronger self-hosted Qwen generation sandbox; Pop closed. Mellum actual7759 progressing, runtime4bda74b8c99a, active26907newtrain+169held (adoption is not exposure). LF hostOOM after6654: simultaneous CPU resident index materialization plus trained-parent prefix qualifier exceeded host RAM; last immutable6620 parent retained, unsaved34updates not credited. LF restoredactualSGD6623/GPU100%11:22:25,native649181; a secondOOM killedCPUadopter649481 duringrestore, nowALLLFadopter/qualifierjobsstopped. VMactualMemAvailable11.44GiB vs misleadingcgroupfree31.6GiB; existingBig2CPU3.88GiB needsaccounting. VM-wideMemAvailableguardandexclusivecheckpointedhandoffrequired. New LF actual three-tier capacity94236physical/157075KV certified01386db6c736592e6d69584a39b48d3414a126d65e5949a6522ee8e8e4e07bdd;75700train+156held append pending actual safe adoption. All previous pending quotas survive. NVIDIA CSF teacher switch ACTUAL: oldengine drained/cancelled at564finished/6interrupted/454unstarted inbatch14, frontier preserved; pinnedvLLM actualCSF startup failed; automatic oldSGLang rollback691016 loading on samegenerationVM, exact frontier retained. No throughput claim. Persistent guardian actualactor checks and drain armed. LFM trained-parent prefix candidate8fc03eb2c250 passes64focused/quick, qualifier051b4cc88 preserves actualtrainedcheckpoint; H1006620actualqualificationPASS:13431to8188physical(-39.04%),bitwisehidden/logits/loss,inputgradrelL2 4.14e-6; no isolatedthroughputclaim. PrefixstillOFF untilsafeadoption. Hostheavy jobs now serialized, no falseclaimGPUoptimal. SourceindexASTcontinuation399162917+182a9ecfd integrated,published,16focused+quickPASS,independentreview complete; actual3042to8fc policyproofPASS. Tool preservesexactentries/sourcekeys, noadmission/exposure. File-backed non-reentrant checkpoint input storage integrated a7f7bc66b+72957ded9 with exact strides/repeatedunpack/storeleases,69focusedPASS and independent actualnativeCUDAHVP43PASS14.60s; no actualcloudlonggraphcapacity or productionenablement. Frozenfilecandidate248cd0924 has34focusedPASS7.10s,6runtimefilesonly, actualcloudqualificationstillpending. Order2 reference arithmetic d2f9d1baa+781180f4d validated, publicnatlang.grad completecapture/derivativeartifact path still agentwork; capabilityguard remains. D24full147985SFT+12413preferencepairs normalized; ZIPbytecontentmultiset proof62e918569. D24v3d boundedCPU466581 convertingMtwinafterverifiedLFview, correctedlauncherheldSHA; bothheldactual==expected, incrementalmemberadmission under existingfullprojection/source/token verifiers; no admittedD24/exposure claim yet. SWEexactchildwitness proves one parent→childrequest edge, asyncresult is not fabricated childanswer. Offlinepreference objective/interleave/sourceproof integrated; actualpaircapacity/stageadoption pending. Original26312communicationfoundation incomplete BOTH, noVAEreadiness or traineddialect claim. Mellum7500heldproduceopenP.00111 despite realFP32updates; diagnosing samecohort history. Consumeclosezero supervision intentional forced marker afterimport without ownedpredecessor. Full74e47f187 GREEN all11steps, standing72957ded91 GREEN all11steps exit0; current399162917 fullcheckrunning (itsredundantSHAwrapperratchetfixed182a9ecfd; rerunneeded). Localcompletedpytestfixtureprune20500574208bytes (6deadOct13-15roots,noopen/mapped/cwd/repopins) executingunderledger. HFmanualD09pending403,next12:45UTC throughOct19. Continuoussourcepreparation/Searchadmission, protectedsplits/fulltrajectories retained. No whole-recipe/qualitylift claim.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `completed-pytest-fixture-audit-root-20261016` | low | 1.0 / 0.0 |  |
| `completed-pytest-fixture-prune-root-20261016` | low | 1.0 / 0.0 |  |
| `cw-lfm-foundation-ferry-2264` | low | 1.0 / 0.1 |  |
| `cw-teacher-frontier-live-2516` | medium | 1.0 / 0.0 |  |
| `d24-admission-fullcheck-20261016` | medium | 13.0 / 4.4 |  |
| `natlang-full-check-3991629178` | low | 13.5 / 4.6 |  |
| `natlang-memory-guard` |  |  |  |
| `search-completed-selection-dedup2-20261016` | low | 2.0 / 0.0 |  |
| `search-completed-shards-admission-current-root-20261016` | low | 8.0 / 0.7 |  |
| `searchowl-native-render-optimized-resume-20261016-r7` | low | 16.0 / 4.4 |  |

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

1762 live entries in `training/neuralese_corpora.json` (316 dgx, 1446 pop); 405 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-16T09:26 dgx-codex-01a13659: Adopt live pending obligations and offline objective shared mechanisms
- 2026-10-16T10:00 dgx-codex-01a13659: Deploy adaptive source and restore improvements with exact current-parent binding
- 2026-10-16T10:20 dgx-codex-01a13659: Shared source reader and numerical context 74e47f187
- 2026-10-16T10:25 dgx-codex-01a13659: Combined active and pending recipe handoff 60ecb1505
- 2026-10-16T10:32 dgx-codex-01a13659: Explicit higher-order reference arithmetic d2f9d1baa
- 2026-10-16T11:18 dgx-codex-01a13659: Exact prefix source-index continuation 399162917
