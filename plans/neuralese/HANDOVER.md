# DGX current state

Generated 2026-10-15 09:21 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
This page holds the present only. History is in git (`git log -p -- plans/neuralese/HANDOVER.md`), in the
decision records and in the coordination log. Live processes and the status pages are the authority.

## Read first

- `AGENTS.md`: owner rules (start here)
- `plans/REVIEW_20261012.md`: strategic review and the accepted direction
- `plans/neuralese/NEURALESE_DIRECT_V1.md`: the active training design
- `plans/neuralese/LINE_PARITY.md`: Mellum/LFM line parity
- `plans/neuralese/DECISIONS.md`: recorded decisions
- `plans/MACHINE_COORDINATION.md`: coordination protocol, data sync, storage
- `plans/OWNER_REVIEW.md`: owner review queue
- `plans/MODEL_FACING_CHANGES.md`: model-facing changes and measurement debt
- `plans/NATLANG_PILLAR.md`: weekly NatLang-pillar slot

## DGX status page (this machine)

Updated 2026-10-15T09:21:34.740056Z by dgx-supervisor.

DGX supervision 2026-10-15 09:22 UTC. Owner objective: comprehensive adaptive training of both models through entire production recipe; no base-drift stop or qualification barrier, no warmup-only endpoint, preserve useful optimizer state and share all generalizable changes.
Cloud LFM H100: exact15300 state resume advancing19034 at last check, finite loss/28.1k supervisedTPS, no OOM. Task16000=1/16 (inherited0), no general lift claim. Full18480 checkpoint SHA0c95ff6da3574ac7f8d1aef30611a8d8c30de3ae046ab43e6deefcdf8e522a6f private PUT/independentGET verified. Flash fixed cuDNN host leak. Current task4000/checkpoint440 cadence, completed-task time~22% steady cycle.
Cloud Mellum Blackwell: exact3108 full state SHAe21f1df8312037c625ebb5497478221375362fe4594e60d7ee50477cd91792f2 private durable. R11 FULL1epoch handoff hit old epoch-prefix guard on combined data+epoch change; shared141a4a8462 repaired with17tests+quick. Same allocation v2 bootstrap35032 / actualtrainer37550, light plan finished09:13:27 (22032updates/41378unseen docs), now CUDA model load; Actual new optimizer updates confirmed through3111 after cache HIT and guard repair. No optimizer/allocation reset. Prepared cache exists229390train/30021held/66479targetwindows; helper adding equivalent light-plan cache reuse. Next PH append checkpoint600 (~40min), prewarm CPU windows/plan before stopping GPU.
Data: R11 FULL admitted/staged both sandboxes; current33 all392pending-group scan zero intersections with50956train docs per twin, immutable group-index proof in dgxops/r11-full-current-source-review-proof.json. PH twins972243docs each completed, independent latest-hold audit running prep-r11-ph-frontier-20261015; LFM per-view frontier persisted, Mellum scanning; finalization rereads latest stable registry. Root finalizes registration when terminal proof passes; do not wait WT. Whole future-stage append supported f6fa77ff.
No-half-measures audit found real omission: data_continuation marks entire doc seen after one consumed window, dropping remaining targets/repeated draws.3-window probe after1update loses2remaining windows. Luna implementing occurrence/supervision-aware fix in isolated worktree, coordinating light-plan cache metadata with cloud helper. Keep current startup advancing; heal missing spans at useful checkpoint, preserve3108 parent and source data for exact reconstruction.
RL: shared capture0e83669a and full-policy e1d1eff9 integrated/pushed;81integrated tests+quick passed. Actual full-size LFM and Mellum MuonSR backward/update finite. Real generic task harness LFM4/4 scored0success0infra; Mellum4/4 scored4success0infra. These are setup checks, not trained improvement. Real value graphs: LFMwidth1024/Pgrad72.59, Mellumwidth2304/Pgrad49.79,2vectors/8readeractions, recorded forward delta0 and durable graph roundtrip. Exact serving-step logprobs match; BF16 parallel differences localized, FP32reference6.68e-6, actualimportance ratios unclipped. Shared exact projection restoration c621e1de and ordinary held-window preservation181178d6 integrated/pushed,6projection/graph+17objective tests/quickpass. RL agent finishing real Neuralese collector transport (existing path was text-emulation-only), distributed bundle/controller, SOTA reuse/deployment docs; production RL readiness NOT claimed yet.
E2E: first4 cohort neuralese-e2e-cloud-small-protected-20261015-v2 ADMITTED a1a03a08,4221records340chains1180edges (712train468held), complete graph/current32/quality/decision proofs, root manifest published. Next27 assembly terminal0:147473records11715chains55489edges (45265train10224held); independent verifier exposed original-source chain missing its producer receipt despite a valid later edge. Helper repairing whole-chain exclusion/provenance with assigned original retained, not admitted yet. Remaining two huge sources restoring on sparecore5/nice19 after actual Mellum update; all33 source archives already independently private durable.
External fronts: DeepDive base1818items4826decisions rendered/registered; long recovery1150cases7374decisions registered2517 and rendering2518,48over-cap remain explicit paging work. D40p08/Evidencep09/RightFitlongp02 active, Dolci/Cognitive53/644native shards remote. Storage93GiBNVMe/101GiBexternal. Full main check2e285629 passed all stages (window cache/precision repair included); newer full check asynchronous. Source parent manifests remain immutable, no archived availability claimed as training exposure.

## POP status page

Updated 2026-10-15T08:44:13.014182Z by pop-codex-01a0e480.

Pop 2026-10-15 08:44 UTC: seed52107 closed 99/99, 89 accepted; curated 80 roots/389 decisions after nine holds; source/render manifests 94b13d1a/4c35e456 registered and SHA verified on DGX, each model 389 PH+248 whole. Source review v32 ccb2b9aa, accepted SHA unchanged. Six Luna slots replenished; seed52108 near terminal, seed52109 prebuilt and launching. Storage ~6.7GiB free; 87.8MB redundant intermediates pruned. Pop GPU idle under data-preparation allocation; cloud training DGX-owned. Shared checkout has concurrent edits; isolated publish active.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `audit-r11-current-holds-expansion-2434` | low | 5.0 / 0.5 |  |
| `cw-e2e-unresolved-chain-publish-0917` | low | 3.0 / 0.0 |  |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-deepdive-long-2518` | low | 2.0 / 0.0 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p08-53417` | low | 8.0 / 4.2 |  |
| `render-external-expansion-deepdive-native-long-20261015-v1-p00-54761` | low | 8.0 / 2.3 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p10-55422` | low | 8.0 / 2.1 |  |
| `render-rightfit-recovered-native-2418` | low | 4.0 / 0.3 |  |
| `test-window-continuation-20261015` | low | 3.0 / 0.0 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (63): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1670 live entries in `training/neuralese_corpora.json` (244 dgx, 1426 pop); 314 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap
- `2026-10-15T01:22:52.988891Z-pop-7451`: Seed52079 admitted source for next protected union
- `2026-10-15T01:24:48.414075Z-pop-43f9`: Seed52079 render DGX verified for next union
- `2026-10-15T01:31:38.987626Z-pop-cb1c`: Apply CLINC source hold v5 and new static tranche
- `2026-10-15T01:35:18.879919Z-pop-36bd`: Gemini CLINC 51-52 render DGX verified
- `2026-10-15T01:38:10.184142Z-pop-5192`: Seed52080 source and render DGX verified
- `2026-10-15T01:44:17.336864Z-pop-7a9b`: Apply source-review v6 in next protected union
- `2026-10-15T02:06:11.838191Z-pop-77b5`: Include reviewed recovery corpus in next union
- `2026-10-15T02:07:01.266364Z-pop-4be2`: Recovery twin render now verified
- `2026-10-15T02:10:09.975007Z-pop-b6fc`: Second reviewed recovery corpus ready for protected union
- `2026-10-15T02:21:31.289117Z-pop-e614`: Seed 52082 broad and third recovery corpora verified
- `2026-10-15T02:27:01.137998Z-pop-dedb`: Gemini CLINC 53-54 and fourth repair corpus ready
- `2026-10-15T02:29:01.226263Z-pop-c5a2`: Luna repair overlap scheduler fixed and slots filled
- `2026-10-15T02:30:11.709136Z-pop-3c4d`: Hold first repair v1 source/render: SciFact claim source defect
- `2026-10-15T02:48:40.392891Z-pop-85ef`: Repair catch-up complete through seed73; scheduler fix on main
- `2026-10-15T06:44:56.330095Z-pop-d714`: Source-review v27 forthcoming from seed52099; freeze R11 binding after v27 if feasible
- `2026-10-15T07:22:17.051045Z-pop-7e3a`: Source-review v28 exact hold affects R11
- `2026-10-15T07:32:33.917775Z-pop-47b2`: Source-review v29; exact new item has no earlier Pop admission overlap
- `2026-10-15T07:46:21.523225Z-pop-e374`: Source review v30 current; four Luna seed52103 conflicts
- `2026-10-15T08:23:43.435793Z-pop-cad7`: Current source-review v31; exact task mismatch

## Latest decisions (coord)

- 2026-10-15T08:42 pop-storage-review: Retire obsolete v13/S2 Pop runs; use selected sources
- 2026-10-15T09:05 dgx-supervisor: Shared full-policy Neuralese RL objective integrated
- 2026-10-15T09:07 dgx-coreweave-sandbox: Qualified optional H100 causal conv selector
- 2026-10-15T09:07 dgx-coreweave-sandbox: Combined epoch and data resume repaired
- 2026-10-15T09:14 dgx-coreweave-sandbox: Prepared-window cache reused for continuation plans
- 2026-10-15T09:21 dgx-supervisor: Exact Neuralese projections restored from mixed checkpoints
