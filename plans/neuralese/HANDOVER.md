# DGX current state

Generated 2026-10-15 22:37 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T22:37:00.636358Z by dgx-codex-01a13659.

DGX owns all execution; Pop is closed. Exactly two student sandboxes plus the stronger self-hosted Qwen teacher remain allocated. Train both through the complete evolving production recipe, preserving optimizer/RNG/valid exposure; the initial26,312 communication foundation remains fixed and later cohorts/ordinary CE obligations remain mandatory.

Cloud22:29 UTC: LFM reached3196 on qualifiedcf7 runtime. Mellum safely stopped at3000, preserving full47,802,833,061-byte parent SHA9ab1a8f0f4c31d5430933ced0c3ff6809216eb87c3b530a021c1e17f8cb7e21d, optimizer/RNG/cursor182952/exposure7923. Cadence-only restore child237693 launched22:32:43; actual resumed updates not yet verified. Routine M tasks move to5000 updates, LFM routine tasks use DGX async; held/checkpoint250 and exact-current mandatory foundation/refresh/final checks remain. M old trainer started a long collector after SIGTERM/fullsave; its owned collector was canceled only after exact durable state proof. Published3bfd39626 fixes future stop handling before evaluation and defers qualification when stopping;30 focused trainer tests plus quick passed. Existing cloud numerical runtime remains unchanged.

Both students train learned reasoning and role rows with FP32 masters, actual committed movement telemetry and1:1 priority teacher/communication interleave. Exact optimizer metadata confirms all effective rates3e-6, including sparse tables in torch Muon. Eight-update two-example train-only contrast .003 versus3e-6 showed useful local learning direction, not generalized quality. Production rates unchanged. Root now runs diverse16-update rate comparison on32 distinct admitted training graphs and16 untrained held controls from exact LF2000; baseline active. No diagnostic checkpoint will be saved. Existing named learning-rate/state mechanism must be extended end-to-end before any production rate change.

Current sources244407LF/244408M graphs,13678held/model,46899 genuine communication pairs. Automatic source append guardians prepare514602 additional graphs/model:17048full+338989header+158565whole, preserving existing prefix/credits and optimizer/RNG. Exact-policy CPU index prewarming now uses spare DGX CPUs for whole views while cloud header scans continue. These additions are admitted/staged, NOT counted as trained. HeaderV2 still references six original data files; do not delete them because old proof admission was revoked.

Qwen teacher batch12 latest22:17:1811finished/1655successes,8active/3826queued,94%GPU/90.6GB. V6 complete candidate8639train/model,0held,~90M tokens/model from498 additional successful executions;8651eligible turns minus12duplicates,1183 explicit action omissions. Authenticated text-emulated provenance preserved without invented learned state. Independent dual-model protection/action/parse/token validation is running; not admitted yet. Big2 disjoint1000-chain DGX conversion finished all10382 trajectories/11154edges with zero render omissions:9556/826 ordinary train/test per model,10245/909 semantic train/test; pending independent admission. Full185GB Big2 source stays cloud/private, cloud front mining continues beyond2400chains, Dolci296/644shards last sample. No source is considered fully integrated from its first100chains.

Expanded semantic cohort46899train/10369held is admitted/staged both sandboxes, original foundation preserved. Canonical waiting controller uses readiness-supervisor4820-async-cadence;0models ready/no joint child. Foundation credits latest LF4116/26312 at3000 and M3664/26312 at3000. Actual complete foundation/trainer receipts required. Ordinary CE obligations persist through16-update VAE/control/distillation/RL refreshes, rotating held checks, full held at formal completion.

Shared sharded checkpoints dd4b689d0 passed actual two-port SDK selected transfer/restore and next-update exactness, tinyCPU mechanism evidence only. New96f76572f prepares complete immutable closure archives and requires independent fullGET SHA/size before pin release; root review requested same-consumer release/prepare race serialization before live adoption. Existing student storage unchanged. Canonical outputs are shared/native_shared_stage and shared/native_rl, not extra nested runs directories.

Actual full-size joint FP8 CUDA proof completed22:12:29 on DGXSM121/Torch2.11: VAE112, joint336, distill224 executed grouped projections; both-model control/distill adapters updated. Controlled GRPO replay matched (LF4.3e-6/M0 gap), M1008 executed projections. Peak CUDA72.41GiB;385.9sec includes183.6sec model load. Earlier configured-only FP8 proof was withdrawn. This is mechanism/update evidence, not task lift or cloudNGC2.14/SM120 qualification. Cloud owner adding existing-parent actual-backend qualification without duplicate47GB reload, preserving RNG and unchanged gates.

Prefix sharing stays OFF. Global ordered sparse mask repair restores bitwise full forward. LFM strict full-size input-gradient qualification passes; M full forward also passes in BF16 and actual FP8, gradients fail. Reviewer localized M backward to exact BF16 fan-in cast plus separately measured nonlinear FP8 incoming-gradient quantization: FP8dX versus FP64 dequantized-weight oracle relL2 .025109, FP32 reference1.17e-7. Reviewer owns shared opt-in persisted higher-precision dX experiment with cost/update evidence; current defaults unchanged. Local LFM pass does not qualify actual H100 restored parent.

Historical pre-row-repair LF2750 held collector finished after5185.6sec:0success,7semantic_mismatch,9budget_exhaustion,0unresolved/infra. Existing scorer marks eligible=true,complete=false; valid historical0/16 with budget-censored cases, never current-weight qualification. Seven completed traces show real wrong answers/effects, not transport failure. Separate cloudLF3000 remains UNSCORED due10unresolved; four inspected requests show model-generated producer spans not closing (3EOS/stop,1outputlimit), not a forced template or HTTP failure. Preserve held protection; don't train on these traces.

DGX exporters/async dispatcher active. Standing full suite225b3d5d1f GREEN. Known obsolete empty-FP8-scope test repaired605f459e5; old dd4 suite superseded as incomplete rather than falsely green. Single standing suite bdadc5a203 now past41% Python without failure. Latest report cherry-pick e3a4cd478 quick passed and pushed. Root inbox ACK22:35 UTC; next by23:25. Reclaimed2,029,333,909bytes by deleting only completed sparse-row migration diagnostic checkpoint after exact-path dryrun and no open consumer; LF2000 active parent retained. Latest free space roughly80GiBNVMe/67GiBexternal. Stage own paths only; never owner untracked expansion plan.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cf7-whole-lfm-index-1794` | low | 3.0 / 0.3 |  |
| `cf7-whole-mellum-index-1795` | low | 3.0 / 0.3 |  |
| `cw-native-lfm-periodic-export` | low | 1.5 / 0.1 |  |
| `cw-native-mellum-periodic-export` | low | 1.5 / 0.1 |  |
| `natlang-full-check-bdadc5a203` | low | 13.5 / 3.8 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 4.0 / 0.4 |  |
| `root-current-state-20261016-2237` | low | 2.0 / 0.0 |  |

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

- 2026-10-15T20:50 dgx-supervisor: Shared native learnability repair and teacher interleave
- 2026-10-15T21:14 dgx-coreweave-sandbox: Cloud repaired students live and disjoint Big2 worker assignment
- 2026-10-15T21:49 dgx-codex-01a13659: Shared FP8 dispatch repair awaits actual CUDA execution qualification
- 2026-10-15T21:53 dgx-native-data: V6 captured text-emulation context conversion
- 2026-10-15T22:01 dgx-native-data: Restore authenticated null-type text emulation blocks
- 2026-10-15T22:28 dgx-codex-01a13659: Native stop requests now defer new evaluations and preserve resumable state
