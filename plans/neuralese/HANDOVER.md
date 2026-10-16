# DGX current state

Generated 2026-10-16 09:24 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-16T09:24:46.088639Z by dgx-codex-01a13659.

DGX 2026-10-16 09:26UTC, continuous supervision ACTIVE. Two training sandboxes plus one strong self-hosted Qwen generation sandbox; Pop closed. Owner latest: bring in ALL deferred improvements; AGENTS2b99e189c records active integration and independent per-model progress. Direct root SDK09:23 LF6269/runtime2af640d3e5a3/GPU100%, M7010/runtimea765bd1bea00/GPU83%; real SGD after exact6181/6813 restores, no rollback. Immutable LF6181 SHA5537bd71 records25755visits/916597rows, foundation10985/26312; M6813 SHA7a8765fa records21070visits/949801rows, foundation8237/26312. No foundation complete/no trained shared dialect claim. Learners ordinary native AdamW FP32 .0003 interface; LFreasoning.0003/Mreasoning3e-6. Actual LFcap92606physical/157075KV, Mcap112008/173589. D12/D16/D18v2 full-row capacity promotions registered, +1739LFfit graphs; D15full+capacity admitted (LF16300fit/118long,M606/5), V17registered8309/model and BOTH staged. Staged/admitted is NOT exposure; prior pending quotas retained. Six-ref D13/D14/D19/D32/D36/D37 full normalizer ACTUAL runner57792109:16:53; existing source converters continue. D20/D24 full acquisition receipts verified; D39 121files42.68GB sizecomplete/fullSHAreceipt underverification, D09manual403, HFwatch next09:45UTC untilOct19. D12factual1993views newJSON live48/48, materialization/export running before fullconversion/admission. Shared resident pending adoption + precommitwithdrawal fix integrated6b3f84c40/13b64b6e1, root109testsPASS; offline reference-free pair objective6d90ab732 integrated, actual source/capacity/mandatory-interleave still being completed. Provider exact sidecar closure183cd6a95 root17PASS; capacity metadata continuation adds resident only after independentreview,29PASS. Native serving adapters actual40CUDA/typedHTTP/restore testsPASS,323680aa8 published; noorder2/fullsizecloud/taskliftclaim. HTTPtransport candidate48/48exit0,45accepted/0infra; baseline47/48lastcase; finish paired receipt then rearm corrected currentstudenttaskeval/controller. Teacher generation continuous; NVIDIA33files102262310289B fullSHAverified, pinnedOCIprep active samegeneration sandbox, engine trial after paired measurement/drain. LFprefix/fullruntimecomposition and bounded checkpointprefetch active cloudowner; Mgradientproblem retained explicit. Searchr7live3NVMeSQLite workers, durableoutputs external; root15GiB/ext52GiB lastsample. Fullcheckbfdc7e75 COMPLETE GREEN includingCUDA/TS; currentmain quick+nextstandingfull required. No goalcompletion/RLlift/entirerecipecompletion claim.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-foundation-ferry-2264` | low | 1.0 / 0.1 |  |
| `cw-metadata-monitor-1792142669` | medium | 1.0 / 0.1 |  |
| `cw-next-admitted-prewarm-2382` | low | 1.0 / 0.1 |  |
| `dgx-d12-status-export-verify-20261016` | low | 2.0 / 0.0 |  |
| `dgx-d12-status-export-verify2-20261016` | low | 2.0 / 0.0 |  |
| `natlang-full-check-323680aa8e` | low | 13.5 / 4.1 |  |
| `natlang-memory-guard` |  |  |  |
| `search-completed-selection-dedup2-20261016` | low | 2.0 / 0.0 |  |
| `searchowl-native-render-optimized-resume-20261016-r7` | low | 16.0 / 4.3 |  |

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

1755 live entries in `training/neuralese_corpora.json` (309 dgx, 1446 pop); 398 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-16T04:00 dgx-codex-01a13659: Seven more acquired source adapters prepared
- 2026-10-16T04:13 dgx-codex-01a13659: Same-forward held boundary diagnostics by graph class
- 2026-10-16T04:18 dgx-codex-01a13659: Read-only held-shape capacity coverage supported
- 2026-10-16T04:34 dgx-codex-01a13659: Shared coefficient RL route integrated; provider mode stays disabled
- 2026-10-16T05:06 dgx-coreweave-sandbox: Native routine evaluation: one deterministic full pass
- 2026-10-16T08:34 dgx-codex-01a13659: Pending-only recipe handoff fix 9cf6c95eb
