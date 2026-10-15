# DGX current state

Generated 2026-10-15 11:54 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T11:54:15.139961Z by dgx-supervisor.

2026-10-15 11:55 UTC — DGX supervisor; Pop owns Pop, coreweave helper owns both cloud allocations.
Owner requires complete native shared-state Neuralese replacement in training and inference, no prototype shortcuts or architectural half-solutions, a separate independent integration review before cutover, full production recipe for BOTH models, and automatic shared semantic VAE/joint producer-write-out/consumer adaptation as soon as TWO distinct models are ready. More models join later; losses/curricula/data/optimizer continuation remain adjustable. Old cloud projected-value jobs continue DIAGNOSTIC ONLY during implementation. AGENTS.md and plans/neuralese/SHARED_STATE_MIGRATION_20261015.md are authoritative; old direct-v1 documents describe diagnostic lineage.
Implementation: shared graph executor uses actual four producer/consumer role tokens, identity-initialized low-rank token-embedding transform, positional raw KV and finite-convolution state, one layer-major graph forward, no recursive producer replay. Native block/provenance/checkpoint factory published. Full actual LFM FP32 graph/reference, consumer CE/source-adapter gradients and incremental capture/splice pass; published destination CLI133f2ce0 also performs real channel optimizer update/exact channel+optimizer restore (NOT backbone recipe/task qualification). Mellum BF16 numerical repair still in progress: shape-dependent linears, attention tiles, fanout adjoints and possible order/cache effects; no tolerance-widening/qualification claim. Pop CUDA parity failures forwarded to graph owner.
Root graph objective47a5690f has whole-trajectory weighted CE and chunked readout, native schema/provenance and boundary checks. Shared semantic VAE core predicts target-state deltas, explicit token alignment, differentiable source-to-consumer joint gradients, stable add_model parameter names. Nine focused tests passed; NOT yet executable cross-model stage. Root owns role-aware full-corpus conversion/admission, cross-model stage/readiness controller and RL replay migration. Operations helper now owns real shared native SFT trainer, optimizer/FP8/checkpoint/RNG/exposure/task cadence/recipe integration for BOTH models. Independent reviewer not yet started: must review integrated complete implementation after ready.
Serving/store/HTTP/TS native harness integration cherry-picked through5dffd64e;24 integrated Python focused tests and quick check passed. Native artifacts have disjoint nzs1_ content IDs, strict inventory/provenance validation, fsync-before-publication durability, same store lifecycle. Real HTTP producer/capture/consumer fixture passed; nested historical tool argument artifacts restored without corrupting JSON, raw native RL sampling receipt survives collector. Guided native serving and learned cross-model decoding remain explicit integration gaps, not qualified complete. Root native RL archive extension detects prompt/write/JSON-argument refs, verifies and copies native artifacts before actor shutdown;6 focused tests passed. Native full RL objective still pending; legacy GradSession not compatible.
Cloud health11:45UTC: Mellum actual trainer65896 step3563 finite/GPU97%,91,195MiB reserved/OOM0, parent full3400 private SHA verified; task3600/checkpoint approaching. Forced layer checkpoints after earlier3418OOM. LFM trainer203166 resumed exact28000 full/RNG under explicit diagnostic-partial evaluation runtime policy; CPU cache rebuild R8 PH target256 (/2), GPU~2GB while rebuilding. Correct PH256 prewarm205061; prior1024 mismatch fixed, wrong WT1024 preparer stopped explicitly. Destination native LFM qualification requested on idle H100 using isolated Git runtime, not canonical overlay. No current task lift claim.
Data: R11 FULL/PH/WT admitted; v36 current source review,398pending groups/zero intersections. FULL next56378docs53223train3155held/twin admitted. REST27v2 147414records137421train9993held11713chains55484edges admitted; all9files14.57GB exact staged both clouds, lazy indexes complete (LFM118.94s/Mellum97.21s). Incremental PH/WT build still running; Pop52110–17+ need final overlay. Search/OWL exact selective6.9GB restored164native turn shards658396decisions; full650GB-decoded streaming hold scan failed because ZERO source-group intersections, not missing row source_groups (schema inspection proved top-level groups present). Need diagnose whether exact zero is correct and produce authenticated full scan proof; admission/render NOT yet complete. Do not blindly demand exclusions>0 or fabricate a hold. 7002 over-cap cases require explicit paged conversion. Big2 protected assembly/Dolci cloud CPU conversions continue under helper.
Generation closeout per owner relayed by Pop: finish existing DGX32-task x2 teacher batch, no new campaigns; document procedural restart handover. Pop replenisher timers disabled, seed52118 final queue thenclose. DGX teacher dgx-native-mellum-teacher32-20261015 remains actual base actor, train-only cohort, reusable source after exact outcome/protection review, NOT trained cloud checkpoint or student on-policy. Curated52116/17 source/render SSH SHA verified; Sol repair pilot2roots18decisions registered, old contaminated Luna repair batch held.
Broad checks: last completed green7c45c961c7;90480ac35a failed TS LFM fixture CUDA OOM during model.to, no claim green/skip. Standing single full-check service handles newer snapshots; never launch duplicate broad suites. Local focused work ledgered; quick checks passed. Storage monitor normal cadence; reusable sources/active parents/held data retained, owner expansion plan untracked and never staged.

## POP status page

Updated 2026-10-15T11:43:12.250819Z by pop-codex-01a0e480.

Pop generation closeout (owner 2026-10-15): Luna/free-Gemini replenisher timers disabled; no new campaigns. Seed52117 terminal and source/render SHA verified on DGX (74 roots/333 decisions); seed52118 paused for a Pop full gate and will resume only its exact pinned queues, then curate/register/render/sync and stop. Seed52116 source/render also synced (79 roots/348 decisions). Sol medium repair pilot admitted 2 roots/18 decisions and was synced; future repair-worker model pin code pending full-gate triage/publication. Pop full gate currently reports 2 shared neuralese attention parity failures (DGX kernel owner briefed); general Python and GPU-heavy neuralese passed. DGX cloud training and DGX generation resource control remain DGX-owned. Handover procedure plans/GENERATION_CLOSEOUT_20261015.md pending push.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `dgx-native-mellum-teacher32-20261015` | low | 45.0 / 30.0 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.1 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p11-62739` | low | 8.0 / 3.5 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (69): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1695 live entries in `training/neuralese_corpora.json` (251 dgx, 1444 pop); 339 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T10:55 dgx-supervisor: Future-only QAT stages can extend running recipes
- 2026-10-15T11:01 dgx-coreweave-sandbox: Compatible Flex decode tiles for mixed cached attention
- 2026-10-15T11:03 dgx-supervisor: Owner replaces Neuralese architecture; existing runs diagnostic
- 2026-10-15T11:09 dgx-coreweave-sandbox: Frozen trainer imports now outrank canonical cwd
- 2026-10-15T11:33 dgx-coreweave-sandbox: Explicit diagnostic continuation through isolated eval transport failures
- 2026-10-15T11:47 dgx-supervisor: Native replacement review and DGX generation closeout
