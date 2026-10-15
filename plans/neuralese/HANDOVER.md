# DGX current state

Generated 2026-10-15 19:00 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T19:00:24.358357Z by dgx-supervisor.

DGX supervisor owns all execution; Pop closed. Exactly two student sandboxes plus one Qwen teacher. No new allocation, Luna generation, or student self-generation. Continue comprehensive evolving recipe with exact optimizer/RNG/exposure; finite communication foundation is a transition boundary, not completion of all ordinary CE. New admitted cohorts stay mandatory without moving an already declared foundation finish line.

Actual student observation 2026-10-15 18:42:53 UTC: LFM H100 sandbox7ee701ae, trainer312478/recipe312461, step1836 loss1.13516,4388supervisedTPS,GPU86%,23GB. Mellum RTXPRO96 sandbox5ffa592c,trainer146209/recipe146157,step1559 loss0.115014,4588supervisedTPS,GPU100%,63.6GB. Both advancing without OOM. Runtime e4dd6aabd3b0, recipe185954/185955 rows including3404QwenV4graphs each.32768batchtokens/1024chunks,task1000,held/checkpoint250. Prior long task checks preserved partial/unscored/unqualified before exact checkpoint handoff; no invented task pass.

Exact safe exported full-state parents verified by independent private fullGET/SHA: LFM1750,2,027,511,243bytes SHA6e97d601e32f2ad1316601e277c97eeff1046fd3945de3e2f2c6f203ae166df7; Mellum1500,47,761,947,901bytes SHAfe66c38b383ff191f107ca6a13db2e3aadca2aebccf7d60761300ac1a043f404. Periodic exporters active. Preserve2454/1766credited rowvisits and all37old communication rows. No full47GBDGX checkpoint copy.

Next handoff prewarming while students train: c4502012a admitted V3 comm cohort26312train/9896held,26275newtrain tail after exact37prefix; e006813d6 admitted512ordinary recovered decisions/model; explicit finite foundation manifest and shared continuation runtime.512 already SHAverified both sandboxes. Large197MBSDK stream failed, switching to qualified private GCS transfer/fullSHA. No training admission inferred from copies. Active recipes currently lack foundation fields, do not claim new boundary deployed until exact runtime/config handoff evidence.

REST raw source55484edges:36208original+18010missing-edge repair+605actual segmented-token offset repair=54823usable pairs before split/protection/admission,661explicit omissions. V3partial admitted c450;18615repair supplement strict validation in progress. Root derived source135additional exact clean communication edges from161of512reviewed original endpoints; explicit new pair-source decision f482cbcf154e378fb2dea457b1e0caee101ca390b31e5859936f4bd34c80afb1, unchanged messages/targets, original false admission preserved in provenance. Rendering runs root rest27-reviewed-pair-render-20261015; output UNADMITTED pending both real tokenizers/geometry/protection/dedup. Failed/quality/held remaining edges excluded. New cohorts must not delay current foundation handoff.

Teacher Qwen3.8FlashNext RadixArkNVFP4 sandboxdadd0462 server68164 resident, PennyRoyal3.0.1SM120,8collectors.18:47 broaderbatch12:947started/939finished/904exact successes;13semantic/21invalid/1budget. Priorbatch11:763/768.1913casesx2; label invalid_action can include semantically wrong valid finished actions; diagnose actual cases. Only exact successful decisions admitted; salvage independently proven children separately without relabeling failed parents.

Big2 full185GBsource/70656chains/1373067edges privately durable and SHAverified. CPUconversion~800chains/18352edges; first100chunk privately durable/restored9MB,2115ordinarytrain154held/model+2153train150held pairs,0omissions. Data agent validating incremental package; do not wait entire conversion.4nice19workers authorized after durable frontier within existing9CPUH100. Dolci252/644shards at18:42. Local evidence render wrapper remains monitored.

Shared runtime integrated cb9a542b6 exact CE frame/credits carry and finitefoundation,2c37521f4 bounded16updatesperphase all-phase semantic refresh/rotating32held/fullheldat formalcompletion,ddc13f79b actual learned transport binder,ef22ff0bb independent compact-wire learned-delta consumer exactreference certificate,495cd7a27 restored-weight precision qualification. Root validations29foundation/40semantic/4actualwire/14precision/13offset passed. Followup652736dc2 adds standalone semantic-only controller continuation with exact parent publications, interruption resume and no O(N^2)exposure scans; quickPASS, root focused stage+readiness validation running. Teacher/model-facing live measurement debt remains explicit, fixture/control tests do not prove task quality.

Actual SDK RL qualification pending: fixture1490 failed prelaunch exact built-task artifact SHA mismatch across sandboxes, no worker/orphan; isolated exact artifact staging1505. Provider closurebfd7846b0 validates nested ordinary/foundation sources, runtime/base/RNGsidecars. No claim real target CUDA RL round qualified yet. Must finish actual round and stop/restart proof before production RL deployment. Legacy coefficient migration and real order2 kernel gradients remain explicit open blockers, not stubbed.

Broad full check e029 failed four stale sampler fixtures, fixed already in current35ca ancestry; reviewer strengthening real Sequence fixture. Current broad bfd running, no duplicate suite. Last measured free storage NVMe100GiB/external68GiB. Coord root identity dgx-supervisor inbox lastACK18:42,nextby19:32; retired Pop delivery failures do not undo local status. Owner untracked expansion plan and dataagent untracked semantic recipe must not be staged by root.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-native-lfm-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-native-mellum-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-pair-v3-private-stage-1508` | low | 1.0 / 0.1 |  |
| `cw-provider-metadata-tests-1510` | medium | 2.0 / 0.4 |  |
| `native-rest27-pair-repair-tail-validation-20261015-v7` | low | 10.0 / 0.5 |  |
| `natlang-full-check-bfd7846b0f` | low | 13.5 / 10.3 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `root-current-state-20261015-1900` | low | 2.0 / 0.0 |  |
| `root-semantic-continuation-bounded-cpu-20261015` | medium | 9.0 / 0.4 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| native_state_sft | lfm | `native-state-production-lfm-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the lfm backbone. |
| native_state_sft | mellum | `native-state-production-mellum-20261015-v1` | native-state-production-base-20261015-v1 | Shared complete native-state production SFT on the mellum backbone. |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (77): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `native-state-production-base-20261015-v1`, `native-state-production-lfm-20261015-v1`, `native-state-production-lfm-rest-qwen-v3-append-20261015-v1`, `native-state-production-lfm-teacher-comm-append-20261015-v1`, `native-state-production-mellum-20261015-v1`, `native-state-production-mellum-rest-qwen-v3-append-20261015-v1`, `native-state-production-mellum-teacher-comm-append-20261015-v1`, `native-state-semantic-curriculum-rest27-partial-20261015-v1`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1709 live entries in `training/neuralese_corpora.json` (263 dgx, 1446 pop); 353 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T11:55 dgx-supervisor: Native graph objective, VAE core and native serving integrated
- 2026-10-15T12:00 dgx-coreweave-sandbox: Explicit partial-runtime diagnostic continuation retains ineligibility
- 2026-10-15T12:33 dgx-supervisor: DGX generation closed; exact restart procedure published
- 2026-10-15T12:56 dgx-supervisor: Pop closed; continue generation only with own sandbox models
- 2026-10-15T13:41 dgx-supervisor: Shared semantic stage implemented; Pop closed and dedicated own-model generation allocated
- 2026-10-15T14:17 dgx-supervisor: Self-hosted stronger teacher and shared native execution
