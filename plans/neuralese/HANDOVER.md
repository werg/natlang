# DGX current state

Generated 2026-10-15 12:54 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-15T12:54:37.002665Z by dgx-supervisor.

2026-10-15 12:54 UTC — DGX supervisor owns all remaining work; Pop is CLOSED.
OWNER CORRECTION: only Pop generation was to finish. DGX generation CONTINUES using OUR OWN MODELS THROUGH COREWEAVE SANDBOXES ONLY. No Luna/external teacher generation or local GPU actor generation. Existing third-party corpora remain usable. AGENTS.md records superseding ownership. Sandbox agent is actively arranging actual sandbox generation alongside training; no new generated results claimed yet. The mistakenly stopped DGX 32-task x2 Mellum-base queue yielded4successes,4budget/unscored,56unattempted; preserve successes for admission, do not relaunch its local GPU actor.
Native replacement: BOTH training and inference, full production recipes, independent review before cutover. Old cloud runs DIAGNOSTIC during implementation. Full Mellum new fixed FP32 row-delta sparse attention now BITWISE forward/logits/CE/all28layer backward/sourceadaptergrad/incremental; actual QAD LFM unchanged tolerance qualification PASS. Core e16f4e93 integrated as982ab3ad; 41focused+quick passed in isolated worktree. H100 destination vendor-lowering qualification remains open. No task-lift claim.
Shared native SFT trainer/optimizer/task/exposure/RNG/resume/packed graph objective integrated5bd7a976/56d1450c/690da54a; authentic forced text/init capture and sampling/2 included. Typed durable state lifecycle and real child handoff independently tested2architectures integrated79102de1. Root focused integrated tests2629 and quick2630 running. Reviewer continues complete client/view/empty-value migration and audit; no approval yet. Root native RL one-graph actor objective/replay publisheddc78e1ee+50887ec2; actual training-stage/actor CLI wiring remains. Shared semantic VAE differentiable core exists, executable joint/distill/export/serving not yet complete. Root readiness controller3testsPASS, still unpublished and not production-bound. No placeholder-completion claims.
Data agent owns role-aware full-source native conversion, train/test protection and BOTH actual production recipe bindings. R11/REST27 admitted; final Pop52116–18 source/render224roots955decisions verifiedDGX, nextunion admission separate. Search/OWL658396decisions40818programs96087sourcegroups full scan provesZEROprotection/review intersections; compact164part indices and source admitteda8e67c1f.7002overcap pagedconversions assigned. Preserve owner untracked expansion plan. Big2/Dolci sandbox CPU work continues;300Dolciwrong_return omissions need runtime-history/index recovery,3fresh unchanged replays succeeded.
Cloud last12:27: Mellumstep3749GPU98%, finite, task3600=8/16 vsbase13, zeroinfra, full3600privateSHAe97eb68e611a1e4acd03e937306c7f293b198134f094d0b0abb72f3b132bf00c. LFMexact28000restored CPU/window2cache prep, no28001 yet, GPUidle prep; correctPH/WTtarget256prewarm, no stale handoffcontroller. Full28000privateSHA737f37ee855c68830a003e2ac6dc9591dd0ac2319e72f085bdb61f2e60dc8fbe. Sandbox agent now owns fresh health plus generation resource integration. Periodic durable exporters active. Do not overwrite live canonical cloud Python source while imported by diagnostic trainer; isolated runtimes or safe exactPID checkpoint handoff.
Checks: standing single full-check service only; last broadb246196a45RED27files. Earlier test runner discarded pre-TAP error context; fixed bounded diagnostics50887ec2. Representative canonical rebuilt TS29PASS, not proof whole suite resolved. Latest integrated tests pending. Every DGX test/probe/build ledgered, generationlow/traininghigh. No duplicate broad suites. Continue supervision <=50min; no sleep while actionable work remains.

## POP status page

Updated 2026-10-15T12:21:32.040373Z by pop-codex-01a0e480.

Pop generation closeout complete (2026-10-15 12:20 UTC): Luna seed52118 terminal 93/93, 89 oracle-accepted, 71 reviewed roots/274 native decisions admitted; source/render corpora registered and SSH SHA verified on DGX. Seeds52116–18 combined 224 reviewed roots/955 decisions, each with LFM/Mellum render twins; next union/recipe training admission is DGX-owned. No Pop teacher workers active; Luna and free-Gemini replenisher timers disabled; seed52119 absent. Closeout/restart procedure plans/GENERATION_CLOSEOUT_20261015.md, published initial commit bef7edf942 with final update pending. Future repair manifests pin gpt-6-sol medium; ordinary Luna low. Pop full gate found two shared attention parity failures routed DGX graph owner; Pop focused/quick/TS/conformance and most Python passed. DGX cloud training and generation resource control remain DGX-owned.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `cw-lfm-periodic-private-checkpoint-export` | medium | 1.5 / 0.1 |  |
| `cw-mellum-periodic-private-checkpoint-export` | medium | 1.0 / 0.1 |  |
| `natlang-full-check-50887ec266` | low | 13.5 / 7.0 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-task-eval-dispatch` | medium | 1.5 / 0.3 |  |
| `render-d40-bounded-resume-2402` | low | 4.0 / 0.1 |  |
| `render-evidence-bounded-resume-2403` | low | 4.0 / 0.0 |  |
| `render-external-expansion-agentic-coding-native-20261014-v2-p12-66417` | low | 8.0 / 0.7 |  |
| `render-external-expansion-evidence-terminal-native-20261015--p18-68195` | low | 8.0 / 2.2 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (69): `diag-e2e-lfm-u10006-r4-ph-heads-3e-6`, `diag-e2e-lfm-u10006-r4-text-heads-3e-4`, `diag-e2e-lfm-u10006-r4-text-heads-3e-6`, `diag-e2e-lfm-u10006-r5-writer-adamw-3e-5`, `diag-e2e-lfm-u10006-v2stream-b`, `diag-e2e-lfm-u10006-v2stream-c3e-4`, `diag-e2e-lfm-u10006-v2stream-c3e-5`, `diag-e2e-lfm-u10006-v2stream-c3e-6`, `diag-mixed-lfm-u10006-masked-clock-p10`, `diag-mixed-lfm-u10006-masked-clock-p25`, `diag-mixed-lfm-u10006-masked-clock-p50`, `diag-mixed-lfm-u10006-p10`, `diag-mixed-lfm-u10006-p25`, `diag-mixed-lfm-u10006-p50`, `diag-mixed-lfm-u10006-ratio-base`, `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-qad-broad-20261013-v1`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-context-recovery-identity-fix`, `neuralese-direct-lfm-qad-broad-20261013-v1-eval-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-exact-resume-v10`, `neuralese-direct-lfm-qad-broad-20261013-v1-measured-task-eval`, `neuralese-direct-lfm-qad-broad-20261013-v1-pop-discrete-ledger`, `neuralese-direct-lfm-qad-broad-20261013-v1-recovery-handoff-v12`, `neuralese-direct-lfm-qad-broad-20261013-v1-report-recovery`, `neuralese-direct-lfm-qad-broad-20261013-v1-stream-header`, `neuralese-direct-lfm-qad-broad-20261014-r5-data-continuation-v1`, `neuralese-direct-lfm-qad-r8-clean-warmup-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v2`, `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v3`, `neuralese-direct-lfm-qad-r8-ph-coreweave-cadence-20261014-v4`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v5`, `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v1`, `neuralese-direct-mellum-r11-full-coreweave-adaptive-20261015-v2`, `neuralese-direct-mellum-r8-clean-warmup-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-20261014-v1`, `neuralese-direct-mellum-r8-clean-warmup-coreweave-held-resume-cadence-20261014-v2`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`, `neuralese-mixed-lfm-qad-broad-v1`, `neuralese-mixed-lfm-qad-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-lfm-qad-r11-ph-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-chain-coreweave-20261015-v1`, `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1`, `neuralese-mixed-mellum-v4`, `neuralese-mixed-mellum-v4-r5-data-continuation-v1`

## Corpus registry

1698 live entries in `training/neuralese_corpora.json` (252 dgx, 1446 pop); 342 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-13T13:46:42.816033Z-dgx-a69b`: Measured: one e2e arm beside v4 costs v4 ~100% (19 s -> 31-47 s/update); pause arms, use the gap

## Latest decisions (coord)

- 2026-10-15T11:09 dgx-coreweave-sandbox: Frozen trainer imports now outrank canonical cwd
- 2026-10-15T11:33 dgx-coreweave-sandbox: Explicit diagnostic continuation through isolated eval transport failures
- 2026-10-15T11:47 dgx-supervisor: Native replacement review and DGX generation closeout
- 2026-10-15T11:55 dgx-supervisor: Native graph objective, VAE core and native serving integrated
- 2026-10-15T12:00 dgx-coreweave-sandbox: Explicit partial-runtime diagnostic continuation retains ineligibility
- 2026-10-15T12:33 dgx-supervisor: DGX generation closed; exact restart procedure published
