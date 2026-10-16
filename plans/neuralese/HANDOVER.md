# DGX current state

Generated 2026-10-16 10:20 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-16T10:20:13.381666Z by dgx-codex-01a13659.

DGX 2026-10-16 10:18UTC, continuous supervision ACTIVE. Owner latest: integrate ALL deferred improvements; two training sandboxes plus one strong self-hosted Qwen generation sandbox, Pop closed. Actual cloud10:12 LF6493 GPU81%/51.95GB and M7177 GPU88%/54.67GB, both useful SGD; teacher14 181finished/8active/835unstarted GPU97%. M host-OOM interruption recovered exact COMMITTED7000 (48158789273B,21395creditedvisits); no credit for lost uncheckpointed7134 progress. D20 cause was rejected 11.6M-row table buffering, bounded rejection fix4ecd2a791 published. LF immutable6181:25755visits/916597rows,originalfoundation10985/26312; M6813foundation8237/26312. Neither foundation complete/no trained dialect claim. Corrected f01 TS controller ACTUAL rearmed10:09 supervisor433722, exact699fileinventory43e248; paired teacher measurement complete96tasks,baseline46/48,candidate45/48,0infra/errors. Current native rates/interface FP32AdamW .0003both,LFreasoning.0003,Mreasoning/backbone3e-6. Actual capacity LF92606physical/157075KV,M112008/173589. Resident+sequentialmmap IO packages LF3042af96a386/M1d7a8d242eb1 independentreview APPROVED, remote preflightPASS, recipe preflight2436 active before stopping trainers; actual handoff pending. LFprewarm75.7knewfitrows40kgeometrydone10:03; M26907fittrain+169held ready. Prior pending quotas must survive (LF2331+118D15,M488+5D15). D12featurestatus full1993/model admitted ffbd94ecc and BOTH staged verified:LF1488fit/505long,M1613/380; no truncated graphs/earnedexposure claim. D12/D16/D18v2 +1739LFfit promotions, D15full+capacity, QwenV16/V17 and Search0008/9/10 registered; staged/admitted is not trained. Shared Search guardian continuously verifies actual source-bound receipts and admits both views. Offline reference-free pair objective+mandatory postjoint interleave+durable both-model source conversion+all-pair original-token rerender integrated (6d90/23c15/3faedf/2b7); root34integrationPASS; actual D24source conversion/capacity/production adoption remains. D20/D24/D39fullacquired; D39all4formatsqualified. D24zip15JSONarrays vs12JSONL requires content-multiset audit, updated8aab verifier running next; first metadata-only audit did not establish coverage. HF09:45 D09manual403,next12:45 throughOct19. Shared source-reader+per-shard numerical metadata74e47f187 fixes linked zstd and future thread context,27tests+quickPASS. Guardian e2a2 persistent boundary repair+cooperative drain integrated, retry actual-actor strengthening under independentreview, notarmed. NVIDIAfullquant102262310289B and pinnedOCI745feb ready same generation sandbox; engine trial after safe task drain. PrefixOFF, exact minimal old-package integration identified5706256bc+b5cff288d, actualLFH100 qualification pending independently of M adjoints. Order2 reference backend/real public natlang.grad migration in progress, no capability claim. Standing fullcheck0369e9110 complete GREEN all11steps inclGPU/TS/conformance. Searchr7 three renderer workers useful; lastroot14GiB/ext50GiB. Continue all deferred production work; no entire-recipe/RLlift/optimal-utilization claim.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-d24-zip-audit-v2-stage-2439` | low | 1.0 / 0.1 |  |
| `cw-lfm-foundation-ferry-2264` | low | 1.0 / 0.1 |  |
| `cw-metadata-monitor-1792145976` | medium | 1.0 / 0.1 |  |
| `miroverse-zip-array-full-20261016` | low | 16.0 / 0.8 |  |
| `natlang-full-check-74e47f187d` | low | 13.5 / 0.8 |  |
| `natlang-memory-guard` |  |  |  |
| `search-completed-selection-dedup2-20261016` | low | 2.0 / 0.0 |  |
| `search-completed-shards-admission-current-root-20261016` | low | 8.0 / 0.7 |  |
| `searchowl-native-render-optimized-resume-20261016-r7` | low | 16.0 / 4.6 |  |
| `teacher-actor-independent-root-20261016` | medium | 2.0 / 0.0 |  |

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

1759 live entries in `training/neuralese_corpora.json` (313 dgx, 1446 pop); 402 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-16T04:18 dgx-codex-01a13659: Read-only held-shape capacity coverage supported
- 2026-10-16T04:34 dgx-codex-01a13659: Shared coefficient RL route integrated; provider mode stays disabled
- 2026-10-16T05:06 dgx-coreweave-sandbox: Native routine evaluation: one deterministic full pass
- 2026-10-16T08:34 dgx-codex-01a13659: Pending-only recipe handoff fix 9cf6c95eb
- 2026-10-16T09:26 dgx-codex-01a13659: Adopt live pending obligations and offline objective shared mechanisms
- 2026-10-16T10:00 dgx-codex-01a13659: Deploy adaptive source and restore improvements with exact current-parent binding
