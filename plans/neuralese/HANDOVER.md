# DGX current state

Generated 2026-10-15 10:42 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T10:42:47.979221Z by dgx-supervisor.

DGX supervision, 2026-10-15T10:42:47.887656+00:00.
RL setup complete and published: main3c83a197 passed the entire integrated check (Python, Neuralese, GPU, TypeScript, conformance). Controller supports parallel native harness actors, exact behavior/token capture, durable transitive value graphs, shared MuonSR learner, full text/value RFT substages, checkpoint/P/TCP preservation, idempotent resume/publication, and newly admitted tasks after exhaustion. Actual recipe/stage required to inherit quality cadence. Both full-size models passed wired value fixtures and finite full-model updates. LFM actual RFT48updates/full epoch19docs48windows, exact-state resume and republish passed; matched trained tasks remained0/4, no infrastructure failures or task-lift claim. Initial controller is synchronous with parallel collection; async overlap unimplemented. Docs: plans/neuralese/RL_OPERATIONS_20261015.md and training/neuralese/rl-configs/README.md.
Production remains adaptive comprehensive BOTH models through all usable recipe data. LFM >24097, task24000=0/16 (two unresolved traces being inspected). R11PH/mix handoff controller waits for current+predecessor /2 CPU caches and terminal task evaluation while old trainer continues. Mellum OOM3418 lost17updates beyond durable3400; mixed .25 relaunch forced layer checkpointing, both /2 caches hit, two real objective-continuation guard bugs repaired (None quantization unchanged included). Guard-fixed trainer53201 passed objective-continuation checks at10:33; optimizer alignment then exposed legitimate stacked MuonSR grad_norm shape rejection and torch load coercion of saved BF16 momentum/FP32 grad_norm to parameter dtype. Cloud helper is repairing exact shape/dtype/value restore with focused regression before relaunch;3400 remains untouched, no reset/no first mixed-update claim. IMPORTANT: saved continuation remains legacy/1 with end22032; code/cache refresh alone does not upgrade coverage. Cloud helper owns explicit same-data v2 coverage upgrade with conservative historical replay at next useful handoff after first real mixed updates.
FP8 OOM cause repaired603b7e0e: autograd retained quantized expert weights/scales, ~10.34GiB omitted from estimate. Added exact shared term/telemetry/new calibration namespace,26focused+quickpass; no global factor/data/context reduction. Unstarted-stage activationa6fd8b29 published,70recipe tests+quickpass; enables admitted WT on existing LFM lineage. These later changes use normal standing next full check.
Data: all R11 FULL/PH/WT admitted; WT369860train13214held per tokenizer, current v36/global381refs/twin proofs passed. FULL/PH/firstE2E currentv36 review398groups intersection0. FirstE2E4221records340chains1180edges admitted. Rest27v2 exact restored147414records11713chains55484edges independent reauth passed; root streaming current hold/hash audit PASSED; registration active admit-e2e-rest27-2566 hit two origin push races and retries, not yet admitted. Manifest-reuse repair80f47d28 avoids unchanged14.5GB rehash on future retries, exact stat/inventory/entry identity required;5focused and quick checks passed in isolated worktree. Big2 actual186GB raw sources restored+member SHA verified; protected assembly active. Next incremental union adds11new sources and retains all381refs; FULL56378docs/twin complete and independent current-protection/twin audit PASSED, registration queued behind root E2E lock; PH/WT running. Search/OWL native private restore ~6.3/11.38GB, bounded render with compact exact global protection control prepared; no source-admission shortcut.
Storage~81GiB NVMe/79GiB external free. Existing cohort/parent sources retained for actual consumers/repairs. Owner plan remains untracked/unstaged. Coordination and background controllers remain active; production completion not claimed.

## POP status page

Updated 2026-10-15T10:16:02.775056Z by pop-codex-01a0e480.

Pop 2026-10-15 10:15 UTC: six-slot Luna replenisher active; seed52113 97/99 terminal on refreshed bounded-iterate runtime, next seed prefilling. Seeds52110/52111/52112 curated 86/77/82 roots and 492/370/420 native decisions; all source and LFM/Mellum render manifests registered, DGX SSH SHA verified, zero local decontam omissions. Source-review v36 431a8a2ddf holds incorrect Pachelbel father-son label; v35 da35769365 holds ambiguous Randy Legge question. DGX confirmed current v36 zero R11 FULL/PH/E2E overlap. Pop full check green 943efb764c; old frozen runtime and 272.6MB preaudit intermediates deleted. Pop GPU idle under data-prep allocation; cloud training DGX-owned. /srv/storage ~5.9GiB free.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `admit-e2e-rest27-2566` | low | 3.0 / 0.2 |  |
| `check-register-race-root-2572` | low | 4.0 / 0.0 |  |
| `cw-http-idle-control-poll-1004` | medium | 1.0 / 0.1 |  |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-lfm-r11-mixed-handoff-0992` | high | 1.0 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `cw-r11-wt-stage-0968` | low | 1.0 / 0.1 |  |
| `natlang-full-check-a6fd8b2968` | low | 13.5 / 0.8 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `next-incremental-union-20261015-v2` | low | 12.0 / 4.1 |  |
| `register-v2-full-ready-20261015` | low | 2.0 / 0.0 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p10-59677` | low | 8.0 / 3.6 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p13-59849` | low | 8.0 / 2.1 |  |
| `searchowl-selective-restore-20261015b` | low | 1.0 / 0.1 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (66): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1681 live entries in `training/neuralese_corpora.json` (249 dgx, 1432 pop); 325 carry training admission.

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

- 2026-10-15T09:57 dgx-supervisor: Integrated full-check timeout assertion repaired
- 2026-10-15T10:00 dgx-supervisor: v36 source holds verified against active cloud data
- 2026-10-15T10:17 dgx-supervisor: Complete shared Neuralese RL controller published
- 2026-10-15T10:21 dgx-coreweave-sandbox: BF16 direct stages may add declared E2E mixing without QAT
- 2026-10-15T10:26 dgx-supervisor: Fix FP8 memory underprediction from retained quantized expert weights
- 2026-10-15T10:32 dgx-supervisor: Activate newly admitted unstarted stages without resetting lineages
