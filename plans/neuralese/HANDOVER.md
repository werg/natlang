# DGX current state

Generated 2026-10-15 10:56 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T10:56:45.169094Z by dgx-supervisor.

DGX supervision, 2026-10-15T10:56:45.051331+00:00.
Production objective: comprehensive adaptive training of BOTH models on all usable recipe stages/data; preserve useful optimizer state and share fixes. LFM latest cloud report 25356, actively updating (78% GPU sample), task24000 remains0/16. Saved representative held cases inspected on CPU: malformed service pagination/loop, missing field in roster, fabricated proof/world result; no held examples admitted as teaching data. Two transport-unresolved traces are TypeError:terminated. Exact130-second idle-stream control passed, so simple120-second timeout hypothesis ruled out. Cloud helper owns deeper error.cause preservation.
Mellum last durable full checkpoint3400 remains exact parent; prior OOM3418 lost17updates. Cloud helper published7a4bc9652d MuonSR stacked grad_norm shape + exact BF16 momentum/FP32 norm restoration,17 focused tests/quick PASS; recipe336244 resealed and actual trainer59071 passed objective guard10:50:36, full optimizer restore ongoing, no first mixed-update claim. Force layer checkpointing remains; shared FP8 saved-backward-weight estimator fix603b7e0e included. Last host sample no OOM/kills,3.44GB anonymous; supervisor checking restore CPU/I/O progress. Legacy continuation/1 coverage still requires explicit v2 upgrade at next useful handoff, not inferred from cache/code refresh. PH+WT exact files staged both boxes; current/predecessor /2 CPU prewarm before LFM handoff while old trainer continues.
Data progress: R11 FULL/PH/WT all admitted; current v36 source-review delta proofs all3views PASS,398groups/zero intersections. Newly published FULL expansion natlang-target-union-20261015-next-incremental-v2-full-ready-v1 has56378docs/twin (53223train3155held), independent381original+72addition protection/twin audit PASS. Newly published neuralese-e2e-cloud-rest27-protected-20261015-v2 has147414records (137421train9993held),11713chains55484authenticated edges; independent whole-chain reauth/global protection/currentv36 review PASS. Root restarted only old REST27 publication after idle git push-race backoff to adopt80f47d28 verified manifest reuse; full audit retained, queued FULL registrar freed and both cohorts published. Cloud helper owns both-model incremental data handoffs; no training-exposure claim.
Adaptation fix13a0bc50 published: future-only QAT after appending/activating stages no longer falsely changes active warmup; normalized no remaining consuming stages, actual active-QAT changes still guarded. Reproduced red first,71recipe/inheritance tests +quickPASS. Cloud helper owns reviewed FULL-chain recipe candidates currently untracked; root does not stage them. Existing unstarted WT activationa6fd8b29 already published. Latest completed broad check a6fd8b2968 GREEN, subsequent changes follow standing single full-check service.
RL machinery remains published/validated for both full-size models. Root found real rendering mismatch in local48-update LFM RFT diagnostic: text stage defaultprogram-header vs actorfull. Shared fix77284286 explicitly aligns managed actor/text learnerfull, defaultfull and receipt actual prompt source;26focused +quickPASS. Captured GRPO/value coordinates unchanged. Live matched-task improvement remains unmeasured/debt recorded; do not claim LFM0/4 proved matched-prompt contract. Current cloud runs are mixed/SFT, not productionRL. Docs plans/neuralese/RL_OPERATIONS_20261015.md.
Local useful GPU work: Luna ops launched dgx-native-mellum-teacher32-20261015 (45GiB low-priority ledger), actual Mellum BASE actor loading (~26.9GiB),32diverse TRAIN tasks x2 samples, exact task verifiers. Shared native text successes may feed BOTH models; not a trained cloud checkpoint or fake on-policy student. Source/quality/held checks before admission; infra failures unscored. Local intermittent full-check GPU unit fits concurrently under ledger.
Other data: next incremental PH/WT union continues; Search/OWL11.38GB native private archive SHA-verified restored, selective6.9GB extraction being repaired for member-prefix normalization without redownload. Data helper adding global held groups to source/quality whole-program exclusions. Shared one-pass compressed partition/render optimization review caught completed-part receipt omission on resume; agent correcting before deployment. Big2 cloud source186GB member-SHA verified, protected assembly continues. Pop seeds52110–14 newly registered beyond current incremental<=52109; nextoverlay planned. Contaminated older-runtime seed52113 repair child teacher-program:f254daddc41181cd0b4d explicitly excluded on future finalization; current registeredcohorts unaffected. Pop action backlog closed through exact verifiers, no acknowledgement-only closure.
Storage last80GiB NVMe/76GiB external free. Active parents, reusable sources and held data retained. Owner expansion plan remains untracked/unstaged. Production completion/task lift not claimed.

## POP status page

Updated 2026-10-15T10:53:33.154981Z by pop-codex-01a0e480.

Pop: six Luna seed52115 workers active; seeds52113-14 curated and twin renders registered/SSH SHA verified (72 roots/311 decisions and 76/362), handover cf7188b52d. Gateway child-failure classifier 09bc661237 and future snapshot f2aa4971c7. Repair cycle s52106-s52113 has one contaminated gateway handoff held from training/DPO; six of ten attempts terminal, four pending. Cloud training DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-lfm-r11-mixed-handoff-1008` | high | 1.0 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `dgx-native-mellum-teacher32-20261015` | low | 45.0 / 28.7 |  |
| `natlang-full-check-7c45c961c7` | low | 13.5 / 0.7 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `next-incremental-union-20261015-v2` | low | 12.0 / 4.8 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p10-59677` | low | 8.0 / 4.4 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p14-61326` | low | 8.0 / 2.2 |  |
| `render-parts-spool-quick2-20261015` | low | 4.0 / 0.0 |  |
| `render-parts-spool-quick3-20261015` | low | 4.0 / 0.0 |  |
| `render-parts-spool-resume3-20261015` | low | 2.0 / 0.0 |  |
| `searchowl-selective-restore-fix-20261015` | low | 2.0 / 0.1 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (68): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1687 live entries in `training/neuralese_corpora.json` (251 dgx, 1436 pop); 331 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T10:26 dgx-supervisor: Fix FP8 memory underprediction from retained quantized expert weights
- 2026-10-15T10:32 dgx-supervisor: Activate newly admitted unstarted stages without resetting lineages
- 2026-10-15T10:43 dgx-supervisor: Registration push-race fix published
- 2026-10-15T10:43 dgx-coreweave-sandbox: Preserve MuonSR stack norm shapes and state precision on projection additions
- 2026-10-15T10:49 dgx-supervisor: RL text prompts aligned; REST27 corpus admitted
- 2026-10-15T10:55 dgx-supervisor: Future-only QAT stages can extend running recipes
