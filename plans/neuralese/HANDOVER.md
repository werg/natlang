# DGX current state

Generated 2026-10-16 12:08 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-16T12:08:53.702543Z by dgx-codex-01a13659.

DGX 2026-10-16 12:08 UTC. Continuous supervision active. Entire evolving production recipe on both models; exactly two training and one stronger self-hosted teacher sandbox.

Training: LFM drained at 6711 (parent e764d30ace98888f25b5b8175bb158a92b4ddb1167f8f1db03088c06e74e69c3), continuation plan 992297 train / 25955 held / all2449 pending. Restore failed before SGD because optional memory_qualification=None was serialized as null. Shared8c50461a1 omits absent setting; frozen57d5818dd283 differs from3042 in EXACT two recipe lines. Root independently verified251 modules/10recursive evidence hashes and continue_capacity PASS, approved review06422d60eec41cf55f5854f41444509c07724345953ea2f19331bde08ff2ade0. Cloud remote verifier PASS; corrected native-recipe supervisor2552 restoring exact6711. No new model measurement or lost new SGD. Keep CPU preparation off LF restore and use actual VM MemAvailable, not misleading cgroup limits. Mellum last observed8146 GPU98%, runtime4bda74b8c99a. Original26312 communication-foundation exposure incomplete BOTH; no VAE readiness claim.

Data: D24 first twin source acquired-d24-sft-member-00372b569325-20261016-v1 registered8c48a8974. Independent strict parse6187train sessions/model,65.25M LF /70.90M M tokens, sameID/groups, zeroheld/quality/pending overlap; allfit current limits. Mellum guarded stage passed31files/114144897bytes; bounded resident append next, not yet exposure. Shared07bb897a6 source pool reuses exact validated subtrees with file hash checks,44focused+114regression+quickPASS; cloud independently reviewed. FullD24 remaining147985SFT/12413preference pairs progressing remotely v3e, real grandchild496285 healthy. D20 metadata-only Parquet accounting52e991264 ready; guarded retry queued after D24 host capacity. Six reference exports require source-owning VM split; retries remain tracked. Search shard0015 registered321b08f1a. V18 transfer853verified-success trajectories (304batch13+549batch14 excludingV1-17), cbf796b6ca22097fb346b3620ed1e99dc465b79b73d122e39e3d0008339edd1e,1.629GB; active DGX8GiB low-priority materializer. No download/staging counted as training exposure. HFaccess next12:45UTC throughOct19.

Teacher: existing qualified RadixArk Qwen rollback actor restored after pinned NVIDIA CSF trial2 loaded74.48GB but ViTFA2 unsupported PTX. io_uring issue fixed by shipped InstantTensor MMAP, same quant. Root inspected actual pinned source; both --language-model-only and --skip-mm-profiling explicitly bypass observed unused multimodal profile path. Third same-image/CSF/MMAP text-only attempt supervisor714768 armed, drains actual case frontier; ready48TRAIN quality/throughput watcher and automatic qualified rollback. No fourth sandbox. Preserve batch14 oldactor564 completions separately from460continuation cases/new actor.

Evaluation: corrected dispatcher and Mellum ingester active after old inactive services repaired. LFM6620 exact parent8aed972119025f4bbc8d218fb15399b7733c290d8eb245ee42d690e0be2fc352 finished16tasks:1pass,6invalid_action,1budget_exhaustion,8semantic_mismatch; eligible, no matched previous baseline. QAD matched base evaluation expressly not requested by prior owner correction. LFM6711 evaluation now active. Root found remaining assistant-only result provenance bug: four actual producer blocks flattened into opaque-ID string returns despite earlier tool-return fix. Repair underway: preserve original wire parts and reuse existing typed return_result path, no payload decode; literal strings unchanged. Next live48TRAIN qualification required before production runtime adoption. Wrongfield and other answer errors remain genuine diagnostic work. Mellum7750 inference view FULLPUT+independentGET verified11:59:30, SHA d8792efed669717c85821edb38523d9f7184bd3e2945ad5328b88deefa64ab2d /24761429633bytes; full origin47a80bbfdb19c7fc014b2e621efbf643ed66c5905ed0ee76d486b2e47f64d8d5. Corrected ingester ferrying to external disk for actual evaluation.

Native learning: e4/6b/0957 independently reviewed and integrated f861e2e61,b802238f2,75a44423d, published c4efd7950. Immutable finite derivative programs; exact state update and prior named moments recomputed through shared CPU optimizer with explicit runtime binding (no crossCPU/CUDA bitwise fiction); full mixed dependency closure across HTTP/storage/.nz/RL admission/objective/restore. Root actual CUDA31PASS+2skip, remaining5 TSbridge PASS after required build, quickGREEN. Public captured-turn objective lowering remains active agent work; public order2 guarded and no full-size RL lift claim.

Efficiency/checks/storage: LF actual H1006620 prefix proof PASS13431->8188physical rows(-39%), bitwise outputs/loss and gradient relL2 4.14e-6, still OFF pending adoption. Source-index continuation399/182 passed16checks and exactfrozen policy verification. File-backed checkpoint candidate248cd0924 has34focused and actual nativeCUDA43 derivative checks PASS; actual longgraph cloud capacity/throughput still pending. Full99da4b04ac GREENall11steps; standing fullc4efd79502 now running. Removed20.50GB dead pytest fixtures and23.42GB discarded legacy projected export with indexes/symlink; active parents/source/held retained. Root29GiB and external71GiB free before current M24GB download. No whole-recipe completion, learned shared dialect or optimal utilization claim.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `corrected-mellum-eval-ingest-root-20261016` | medium | 3.0 / 0.4 |  |
| `corrected-native-task-dispatch-root-20261016` | medium | 3.0 / 0.6 |  |
| `cw-lfm-foundation-ferry-2264` | low | 1.0 / 0.1 |  |
| `cw-mellum-d24-sourcepool-prepare-2554` | high | 1.0 / 0.1 |  |
| `d24-v3e-progress-20261016-04` | low | 1.0 / 0.0 |  |
| `natlang-full-check-c4efd79502` | low | 13.5 / 0.9 |  |
| `natlang-memory-guard` |  |  |  |
| `search-completed-selection-dedup2-20261016` | low | 2.0 / 0.0 |  |
| `search-completed-shards-admission-current-root-20261016` | low | 8.0 / 0.7 |  |
| `searchowl-native-render-optimized-resume-20261016-r7` | low | 16.0 / 4.5 |  |

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

1765 live entries in `training/neuralese_corpora.json` (319 dgx, 1446 pop); 408 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-16T10:20 dgx-codex-01a13659: Shared source reader and numerical context 74e47f187
- 2026-10-16T10:25 dgx-codex-01a13659: Combined active and pending recipe handoff 60ecb1505
- 2026-10-16T10:32 dgx-codex-01a13659: Explicit higher-order reference arithmetic d2f9d1baa
- 2026-10-16T11:18 dgx-codex-01a13659: Exact prefix source-index continuation 399162917
- 2026-10-16T11:24 dgx-codex-01a13659: Deferred integration and completed-fixture cleanup 99da4b04a
- 2026-10-16T11:46 dgx-codex-01a13659: Retired resident pointer and teacher continuation source fixes
