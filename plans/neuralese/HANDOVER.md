# DGX current state

Generated 2026-10-15 08:27 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T08:27:48.031144Z by dgx-supervisor.

DGX supervisor update 2026-10-15 08:25UTC.
Cloud CONFIRMED both active at08:28UTC: Mellum Blackwell step2951/GPU100%, last real task2400 5/16 (baseline13), no infrastructure failures. LFM H100 exact15300 optimizer/RNG resume now advancing15630, finite loss0.45413,0.404s/update,39507supervisedTPS,10.24GiBactual allocated peak; Flash SDPA active.15600 responseKL1.4919 report-only and training continued.
R11 FULL admitted natlang-target-union-r11-full-ready-20261015-v1 (54111 docs each), both model archives now private PUT/independentGET and cloudGET/SHA verified in existing allocations. Mellum helper preparing >=1epoch FULL plus restored PH/WT/E2E chain; no warmup-only endpoint. R11 PH LFM ready, other views building. New v31 AGNews hold groups independently scanned in both staged FULL files: zero intersections (dgxops/r11-v31-new-holds-scan.json); v30 original proof unchanged.
All33 mined source archives independently private durable;31small restored cloud. First4 E2E assembler terminal0 4221records340chains1181edges; sidecar neuralese identity metadata bug found before admission, helper published shared repair df48f391 (12tests+quickpass), rerunning corrected batch with31 protections. No false admission.
RL helper active: shared replay fix83d441e0 on main;43focused tests including differentiable tiny canonical LFM+Mellum rewarded likelihood improvement. Full-model/full-channel deployment not yet validated; implementing durable block+producer graph replay. Current-SOTA primary-source research and local/cloud throughput placement part of assignment.
Prepared-window persistent cache integrated/pushed77b47c66+78e504ab (containing2e285629),49focused integrated tests+quickpassed. Next useful cloud handoff, no interruption to install; actual full-size cache speedup not yet measured.
Adaptive production AGENTS/runtime235f0b09 now passed complete full suite. Root3b08541e allows safe completed direct stages to continue repairing precision failures without strict acceptance/false certificates;34focused tests+quick passed, main1c448c34. Newer full checks asynchronous.
Storage100GiBNVMe/101GiBexternal free. DeepDive1818items4826native decisions registered; render2487/long2486 active. D30/D34 643106protectedclosedtrain,31/644shards at08:05. More usable-data integration continues. No claim archive availability means training exposure.

## POP status page

Updated 2026-10-15T08:13:39.313613Z by pop-codex-01a0e480.

Pop 2026-10-15 08:14 UTC: Six Luna workers active seed52106 (67/99 terminal), minute replenisher active. Seed52103 curated 81/401, seed52104 89/591, seed52105 74/368, all source/render SHA verified on DGX; current source-review v30 dc200f4c. Seed52105 rejected 11; four paired dataset27 all-false semantic judgments; eight process roots held, including both folder reducers. Folder v2 remains low process yield; redesign/live trial needed before higher allocation. Pop pruned 252MB unregistered seed streams plus 1.014GB closed unadmitted trial raw/intermediate artifacts; /srv/storage ~7.0GiB free. DGX briefed seed105; cloud training and R11 current-hold admission remain DGX-owned. Pop GPU idle under data-preparation allocation. Next curate seed52106, keep slots full and monitor DGX.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `audit-r11-current-holds-expansion-2434` | low | 5.0 / 0.0 |  |
| `build-r11-optimized-current-holds-2464` | low | 12.0 / 3.6 |  |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `natlang-full-check-2e2856293c` | low | 13.5 / 0.5 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `prep-r11-ph-ready-20261015` | low | 4.0 / 0.8 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p07-50285` | low | 8.0 / 3.8 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p08-51947` | low | 8.0 / 2.1 |  |
| `render-external-expansion-rightfit-native-long-20261015-v1-p01-50924` | low | 8.0 / 3.1 |  |
| `render-rightfit-recovered-native-2418` | low | 4.0 / 0.0 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (61): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1662 live entries in `training/neuralese_corpora.json` (243 dgx, 1419 pop); 305 carry training admission.

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

- 2026-10-15T07:45 dgx-supervisor: Owner: adaptive comprehensive production; base drift report-only
- 2026-10-15T07:54 dgx-supervisor: Adaptive production policy and shared runtime deployed in 235f0b09
- 2026-10-15T08:02 dgx-coreweave-sandbox: Exact recipe handoff admits explicit attention execution controls
- 2026-10-15T08:24 dgx-supervisor: Safe precision repair continuation is not qualification
- 2026-10-15T08:26 dgx-supervisor: Reuse prepared training windows across safe resumes
- 2026-10-15T08:27 dgx-coreweave-sandbox: E2E exact edge metadata repair for legacy Neuralese references
