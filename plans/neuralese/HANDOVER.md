# DGX current state

Generated 2026-10-15 20:50 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T20:50:00.223563Z by dgx-supervisor.

DGX supervisor owns all execution; Pop closed, SSH peer null (queued messages retained). Exactly two student sandboxes plus one stronger Qwen teacher. No new allocations, Luna generation or student-self-teacher. Comprehensive evolving production recipe; preserve full optimizer/RNG/valid exposure. The fixed 26,312 communication foundation is a transition, not completion of the ordinary CE recipe.

Cloud last measured 20:36 UTC: LFM H100 7ee701ae step2681 loss.287,4929 supervised tokens/s,GPU98%; Mellum RTXPRO96 5ffa592c step2375 loss.170,4629 supervised tokens/s,GPU99%. Current runtime d3b401e1b34a,212170/212171 train rows,13051 held;571 invalid historical targets/model withdrawn,5908 valid old rows retained in exact order. Consumed bad exposure remains recorded. Batch32768/chunk1024,task1000,held/checkpoint250 plus3600s checkpoint timer.

AUTHORIZED combined repair+all-admitted data handoff is pending pre-stop plan-only success, with old GPUs kept training until then. Candidate244407 LFM/244408 Mellum train,13678 held each;32237 new train and627 new held, exact old prefix. All source bytes/indexes staged/warm both. Includes9535 QwenV5 ordinary,5 typed-reference pairs+2held,18294 pairrepair+321held,135 reviewedpairs,2115 Big2ordinary+154held and2153 Big2pairs+150held. Foundation remains26312. Guarded bf16x6 LFM requested with actual restored-weight IEEE parity gates and IEEE fallback; no tolerance relaxation. Failed v1/v2 preflight stopped no trainers; 947ef2d8c fixes scoped precision operational tuning allowance, cloud instructed regenerate v3 and proceed automatically. Exact saved parents must be at least last observed live step; no rollback to old exports.

Confirmed two learnability defects, repaired and independently reviewed: QAD thinking-token frozen rows almost identical, ANY-backbone pair logodds bounded.007926; BF16 learned sparse rows lose production-rate updates despite nonzero gradients. cb6e5136a adds learned existing reasoning-token overrides, d327276e6 FP32 masters for ALL sparse rows with unchanged activation casts; cd944ae6a explicit legacy CEframe precision adoption. Explicit migration preserves numeric parameters/moments,counters,RNG,old valid exposure; only new reasoning parameters have fresh moments.5b967510a logs actual per-token gradient/update norm and changed coordinates after real optimizer steps. Full QAD2000 CUDA qualification terminal PASS:149 old optimizer histories exact,initial full logits/embeddings/RNG exact,16 admitted V5 train updates loss.625005 to.268292,pair capacity.043462,exact full checkpoint+optimizer reload. Final first-think p still.000549: mechanism proof, not quality qualification. Actual fullsize Mellum first-update verification remains pending cloudcutover. Native/stockHF logits and18case exact optimizer audits ruled out multiple alternative causes. AGENTS guidance updated pending commit.

Full private GET/SHA verified exports: LFM2500,2053480587B SHA51c76cfee70c0f38125b1574b9597182f957f31637be120ff04303ea639f6d7b; Mellum2000,47787815997B SHAc6aba741f8bc6341035cd5cf4e6e0ab63c156940466fff7346c25e9b6a3878fc. Both periodic exporters active. Local LFM2000 is assigned active before/after diagnosis; no47GB Mellum localcopy. Root diagnostic upgraded2GBcheckpoint remains assigned repair/reload evidence until production validation; then prune exact obsolete path.

Teacher latest20:25:1347finished1249 exact-success,3826queued,8active; Qwen3.8FlashNext RadixArkNVFP4/PennyRoyal3.0.1SM120. Big2 corrected4nice19CPUworkers latest1600chains37111edges, full185GBremote/private durable; Dolci275/644 at20:10. Cloud investigating minimal disjoint CPUbatch offload to spare DGX, no full185GBcopy or newparser. Teacher generation remains fed. All held/quality/pending guards maintained; failed actions contextual only, independently correct child targets retained.

NEW program-header corpus ADMITTED20:47:338989train+299held EACH model, immutable manifest c5e4297cb668fba8f8e78ff45c21ac00d5c68490e77bc3d1d0ff8050d1cf2adf. Exact full file hashes/builder checks,current live held16133/quality65161/pending398 closure,zero intersections; ordered twin IDs/splits/source_groups all match. LFM deep scan passed2825517011 supervised tokens; Mellum deep scan honestly not run, builder row checks passed. Cloud staging next append without delaying current repair. Corrected wholetrajectory v2 active both models (about45%/41% at20:49);597c2e1a7 shared role-aware weighting fixes old unadmitted wholeview's prompt/failedaction overweighting:context.05,tools.25,validassistant1,failedassistant0. Full17048/model converted view admission still needs audit. Never admit old invalid wholeview for parity.

Shared semantic/RL uses exact carried CEframes+remaining quotas through jointVAE/control/distill/RL/back; bounded16updates/phase and rotating32held,fullheld formalcompletion. Initial semantic curriculum exact26312train9896held already staged. Future immutable46899train10369held pairunion provision agreed: after first qualified publication, requestedsemantic_config expands,qualified_refresh_config retains qualified old cohort until mandatory new rawCE catchesup. ActualSDK CPUrefresh76artifacts1400975B all privately ferried; small artifacts now batched via existing sealedarchive. ActualSDK mandatorySFT round and hostTERM stop/restart proof passed (step638/global641,12artifacts1166419B,full optimizer/RNG preserved). Fullsize CUDA RL/joint fit/behavior qualification remain pending. Cloud planning existingsharedcheckpoint participant sharding to avoid47GB cross-sandboxferry; no newoptimizer/protocol. Legacycoefficientmigration andactualorder2derivedgradients remain explicit open work.

RLagent implementing exact-checkpoint native async historical LFM task eval on spare DGX using existing task_eval_dispatch/task_cadence/privateexports; coalesce pending only,never cancel active on newer checkpoint; full checkpoint/runtime/precision identity required, final currentweight synchronous qualification unchanged. Root shared scheduler improvement pending review: alternate communication batches with explicitly priority-positive teacher batches to avoid waiting hours for first positive reasoning supervision, preserve all quotas/foundation;5focused tests pass,actualresume test running.

Broad2d309c67b andddaa2577a GREEN; current single standing suite cd944ae6a. Latestrootprecision2cases+quickPASS andsparserowtelemetry4+optimizer18PASS. Storage20:48 NVMe88GiB/external68GiB. RootcoordACK20:40,nextby21:30. Current state timestamp20:50UTC; live processes authoritative, cloudactualsamples explicitly older above. Owner untracked expansionplan neverstage. Stage only ownpaths.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-evidence-header-stage-1664` | low | 2.0 / 0.0 |  |
| `cw-native-lfm-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-native-mellum-periodic-export` | low | 1.5 / 0.1 |  |
| `evidence-whole-corrected-lfm-20261015` | low | 5.0 / 0.6 |  |
| `evidence-whole-corrected-mellum-20261015` | low | 5.0 / 0.7 |  |
| `natlang-full-check-cd944ae6a3` | low | 13.5 / 3.9 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `root-current-state-2050-20261015` | low | 1.0 / 0.0 |  |

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

1718 live entries in `training/neuralese_corpora.json` (272 dgx, 1446 pop); 362 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T12:33 dgx-supervisor: DGX generation closed; exact restart procedure published
- 2026-10-15T12:56 dgx-supervisor: Pop closed; continue generation only with own sandbox models
- 2026-10-15T13:41 dgx-supervisor: Shared semantic stage implemented; Pop closed and dedicated own-model generation allocated
- 2026-10-15T14:17 dgx-supervisor: Self-hosted stronger teacher and shared native execution
- 2026-10-15T19:55 dgx-supervisor: Verified native deployment and continuation fixes
- 2026-10-15T20:04 dgx-supervisor: Confirmed LFM frozen thinking-token learnability bug
