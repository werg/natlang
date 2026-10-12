# POP current state

Generated 2026-10-12 19:34 UTC by `python3 scripts/current_state.py --machine pop`. Do not edit by hand: regenerate.
This page holds the present only. History is in git (`git log -p -- plans/POP_CURRENT.md`), in the
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

## POP status page (this machine)

Updated 2026-10-12T19:34:00.093852Z by pop-codex-01a0e480.

Pop 2026-10-12 21:36 CEST: Main LFM direct Muon v3 continuation v14 live at step11398+ from full optimizer; 100% GPU, ~7.6GiB VRAM, ~2.0s/update median. Held CE through 11000 remains 1.83-1.92. Objective continuation record starts family/Q4 ramp at step11204; zero mix/member initially expected, family begins after 0.05 of remaining 49,637-update span, Q4 after 0.15. Separate E2E 200-update arm unqualified: held written .471->3.706, crisp .364->2.944, median training loss .032->.00003; all records one decision-chain family. DGX owns stratified sampler and joint backbone+projection stabilization. Six Luna lanes live: CLINC v16 lane03/05 and v17 lane06; MuSiQue E2E seed52001 lanes01/02 and seed52002 lane03 (50 fresh, reference-verified, zero source-ID overlap). CLINC v16 lane04 finished complete. MuSiQue early outcomes 3 accepted/4 answer rejects; no runtime failure pattern yet. Storage root 44GiB free, /srv/storage 13GiB. QAD FP32 base/head anchor pinned for next fresh run; current continuation uses original LFM. Shared code at bca3fd28.

## DGX status page

Updated 2026-10-12T16:22:53.009574Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft on r3: update ~970/31,281 at 16.4 s/update. Ledger reserve is now proportional (fc8a892a). Review workstreams running: e2e progression for both lines (lockstep), task-success gate (e7f09db8), nested+QAT, efficiency, revival (harness/memetic/RL/llama.cpp), housekeeping. One combined v4 handoff is pending those. Memory beside v4 is tight (about 5–8 GB headroom), so small jobs queue.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `natlang-luna-clinc-v16-lane03-v3` |  |  |  |
| `natlang-luna-clinc-v16-lane05-v2` |  |  |  |
| `natlang-luna-clinc-v17-lane06-v3` | low | 1.5 / 0.2 |  |
| `natlang-luna-e2e-musique-s52001-lane01` | low | 1.5 / 0.2 |  |
| `natlang-luna-e2e-musique-s52001-lane02` | low | 1.5 / 0.2 |  |
| `natlang-luna-e2e-musique-s52002-lane03` |  |  |  |
| `natlang-maple-dgx-tunnel-20261005` |  |  |  |
| `natlang-pop-muon-main-v3-resume-v14` | high | 5.5 / 10.6 | neuralese-direct-lfm-muon-main-20261012-v3-pop-launch.json |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (22): `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u10006-pop-launch`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`

## Corpus registry

1372 live entries in `training/neuralese_corpora.json` (115 dgx, 1257 pop); 95 carry training admission.

## Open coordination requests to pop

- `2026-10-11T09:29:18.442129Z-pop-aab3`: V41 enum diagnostic and V40 no-enum bundle ready
- `2026-10-12T16:17:43.543343Z-dgx-f296`: Gate is report-only; task-success eval in the cadence; LFM base 0/48 root cause (1cc33160, e7f09db8)
- `2026-10-12T16:24:16.095493Z-dgx-345c`: Shared e2e lineage on main (983b9fc6): LFM qualification arm launch + handoff path of your LFM main into the shared lineage
- `2026-10-12T17:07:37.233564Z-dgx-1fd4`: Held-task base 0/48: 27 model_incomplete come from scoring the base under program-header rendering (no tool schemas)

## Latest decisions (coord)

- 2026-10-12T17:55 dgx-claude-d8484b5f: Owner: do whatever is necessary to make e2e work well; proceed with your explicit transition
- 2026-10-12T17:56 dgx-claude-d8484b5f: Owner clarification: program-header is an efficiency suggestion, not a rule
- 2026-10-12T18:28 dgx-claude-d8484b5f: Owner: e2e data includes everything
- 2026-10-12T18:33 dgx-claude-d8484b5f: Lockstep: take the windowing change in your LFM handoff
- 2026-10-12T19:19 dgx-claude-d8484b5f: Owner: projection and backbone train jointly; frozen TCP is diagnostic only
- 2026-10-12T19:19 dgx-claude-coreweave: CoreWeave sandboxes: plan + controller (b00b2c94); Dockerfile deps now requirements.txt
