# DGX current state

Generated 2026-10-15 22:06 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T22:04:02.662961Z by dgx-codex-01a13659.

DGX owns all execution; Pop is closed. Two student sandboxes and the stronger self-hosted Qwen teacher remain allocated. Train both students through the complete evolving production recipe, preserving optimizer/RNG/valid exposure. The initial 26,312-pair communication foundation remains fixed; later cohorts and ordinary CE obligations remain mandatory.

Latest cloud sample 22:01 UTC: LFM has resumed from exact step3,000 and reached3019 (loss.2133,6055 supervised tokens/s). Mellum reached2866 (loss.1300,2200 tokens/s,99% GPU). LFM previously exited at21:46:55 because the third consecutive unscored routine task evaluation triggered a legacy fatal counter. Its exact full checkpoint was preserved (SHA fd7932ad14128070fbcd096f9ccf1a4a9f049222363551f96dede5080cc3dfd8); resume retained optimizer/RNG/cursor189974/exposure7575. The recipe-only continuation on qualified cf7 runtime moves routine LFM task checks to DGX async monitoring (task_eval_every=0), while foundation/refresh/final exact-current-weight checks and held CE/checkpoint remain mandatory. Mellum cadence is unchanged. No unqualified FP8 runtime was activated.

LF3000 task evaluation drained after28 minutes: one success, two semantic failures, three repetitions and ten unresolved outcomes; the score is invalid/unscored. Four inspected original requests had no forced write template: the model itself generated a producer span, then three stopped on EOS/server-stop tokens and one reached8192 output positions without closing it. No HTTP failure. Exact final sampled token IDs were not captured by failed streams. A128-pair label audit found producer-open/close supervised at weight1 on both models. Cloud agent is implementing explicit runtime_failure_policy=report for native routine checks, preserving counters and unscored evidence while mandatory boundary qualification remains strict. Legacy default behavior is retained.

Both students run learned reasoning rows, FP32 sparse-row masters, genuine per-token committed-update telemetry and priority-teacher/communication interleaving. Actual repaired updates prove sparse rows move. Stored optimizer metadata confirms all effective rates3e-6, including sparse tables in torch Muon on both models. A train-only eight-update comparison from exact LF2000, using one complete admitted foundation pair plus one V5 row, changed only sparse-table rates to.003: one thinking-close target probability improved .000545→.868; producer-close improved1.63e-9→1.41e-6 and remains weak. This diagnoses local learning dynamics only, not generalized quality. Production rates are unchanged. Reports/source/RNG bindings remain under /home/werg/data/dgxops/native-sparse-row-rate-20261015; no diagnostic checkpoint was saved.

Current training sources remain244407 LFM/244408 Mellum graphs,13678 held per model,46899 genuine communication pairs. Evidence header V2 (338989 train/299 held per model), corrected whole V2 (158565/115), and full view (17048 train) are admitted and being prepared for explicit source/exposure append. Header/whole exact files are staged both sandboxes, not counted as trained. Old header admission was revoked for mutated proof; V2 has separate immutable proof but still references the six original data files. Corrected whole weighting is context.05/tool.25/valid assistant1/failed assistant0.

Teacher batch12 last sample21:43:1665/3826 finished,1532 successes,eight active. V6 has498 additional exact successful executions. Authenticated provider provenance now restores text-emulated block histories without invented learned state or stripped context (f99d3d404,424bae158). Initial LFM output had8636 graphs/90.18M tokens; three authentic type-null cases prompted an unadmitted rebuild for both tokenizers. Data agent owns rebuild, dedup, protection checks and admission. V6 is not yet admitted.

Expanded semantic union native-state-semantic-active-cohort-20261015-v1 is admitted and verified on BOTH sandboxes:46899 train/10369 held, preserving original foundation prefix. Future refresh uses the expanded cohort only after its raw CE obligations catch up. All ordinary CE quotas and full optimizer/RNG survive semantic/control/distillation/RL transitions. Refresh phases use16 updates with rotating held checks; full held evaluation remains required at formal completion. Deployment waits for actual foundation receipts and successful trainer completion.

Participant-sharded checkpoints dd4b689d0 passed root independent code review and actual SDK M→H100 selected transfer/restore. Twenty files/886381B transferred in81.3s; absent peer payload did not prevent selected LFM weights/all optimizer/RNG/CE/VAE restore, and the subsequent update matched source loss/weights/optimizer exactly. This is tiny CPU mechanism/transport evidence, not a trained-parent GPU qualification. Global-last commit, closure pins and unreferenced-object pruning are tested. Future exporters must pin full consumed closure, never only the small global manifest inode. Current student storage format is unchanged.

Big2 cloud mining passed2400 chains/55139 edges. The185GB full source stays cloud/private. A disjoint1000-chain subset was fully extracted, privately uploaded/independently fetched and restored:12MB compressed,1.525GB complete records plus exact provenance. Data agent owns local CPU conversion after V6. First100 admitted chains are not full-source integration. Dolci last sample296/644 shards,22 explicit page omissions.

The original both-model CUDA joint/control/distill/controlled-GRPO run completed with real gradients/updates/replay at72.41GiB peak allocation. It did NOT prove FP8 execution: backend configuration failed to select grouped dispatch. Root repair631a39b2a scopes grouped dispatch plus precision/replay, preserves fixed-row dense CUDA routing, and counts successful projections. Focused11 tests and quick passed. Runner dda4633f0 now rejects zero executed FP8 calls per phase; three focused tests and quick passed. A fresh frozen actual-both-model run is being prepared before any production backend handoff; no artificial foundation certificate or model-quality claim.

Prefix optimization5706256bc remains OFF. Full-size failures were localized to identical Q/K/V and causal histories but different sparse attention FULL/PARTIAL block traversal order, amplifying through Mellum routing. Reviewer global ordered-mask repair preserves exact visibility and sparse GPU kernels; CPU18 and actual CUDA5 tests plus quick passed, including non128 prefix split and cached/import gradients. Full-size paired prefix forward/gradient and cost qualification remain pending. Potential query savings are not an end-to-end speed claim.

DGX historical LF2750 protected task collector remains active with fresh replies; it is pre-repair baseline, not current-weight qualification. Its20GB claim follows measured18–19GB use without restart. Existing async dispatcher coalesces future exports after this collector drains. Both periodic exporters remain active. Full suite225b3d5d1f is now GREEN; the single standing suite advanced to dd4b689d07. No duplicate full suite. GPU probes are ledgered high, CPU conversions/staging low. Root coord inbox acknowledged21:54; next by22:44. Latest storage approximately79GiB NVMe/67GiB external free. Keep current/protected sources and live parents; stage only owned paths, never the owner's untracked expansion plan.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-native-lfm-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-native-mellum-periodic-export` | low | 1.5 / 0.1 |  |
| `native-joint-executed-fp8-cuda-20261016` | high | 84.0 / 32.6 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 4.0 / 0.3 |  |
| `root-current-state-2207-20261016` | low | 0.2 / 0.0 |  |
| `taskeval-bc9f2dba-0002750` | medium | 20.0 / 18.5 |  |

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

1720 live entries in `training/neuralese_corpora.json` (274 dgx, 1446 pop); 364 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs

## Latest decisions (coord)

- 2026-10-15T20:04 dgx-supervisor: Confirmed LFM frozen thinking-token learnability bug
- 2026-10-15T20:50 dgx-supervisor: Shared native learnability repair and teacher interleave
- 2026-10-15T21:14 dgx-coreweave-sandbox: Cloud repaired students live and disjoint Big2 worker assignment
- 2026-10-15T21:49 dgx-codex-01a13659: Shared FP8 dispatch repair awaits actual CUDA execution qualification
- 2026-10-15T21:53 dgx-native-data: V6 captured text-emulation context conversion
- 2026-10-15T22:01 dgx-native-data: Restore authenticated null-type text emulation blocks
