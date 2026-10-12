# POP current state

Generated 2026-10-12 20:34 UTC by `python3 scripts/current_state.py --machine pop`. Do not edit by hand: regenerate.
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

Updated 2026-10-12T20:34:34.693949Z by pop-codex-01a0e480.

Pop 2026-10-12 22:35 CEST: LFM Muon direct v15 resumed full-state step11204 after v14 task eval failed on missing Node and lost 796 uncheckpointed updates. Commit 2a367bef mounts/preflights Node and saves updates before eval; replay-only log tails trimmed. v15 is at step11640+, near 100% GPU and 7.7/8 GiB VRAM; next held/task evaluation at12000. Six Luna lanes active: CLINC v17 lane06; MuSiQue original seed52001 lanes01/02, seed52002 lane03, seed52003 lane05; matched answer-phrase seed52002 lane07 (50 statically verified new cases). Matched seed52001 variant finished 47/48 with 1 resource-exhausted case at 384 requests; 22/47 exact passes and 25 rejected answers under independent semantic review. Original seed52001 incomplete; do not compare unpaired rates. CLINC v16 failed-root exact-child salvage302; Gemini independent review92 successful then 429, NVIDIA Nemotron3.5 Lightning reviewing remaining210 with valid typed output. None admitted. E2E u10006 still unqualified; DGX owns sampler/stabilization. DGX requested full check for 2a367bef. /srv/storage about17GiB free.

## DGX status page

Updated 2026-10-12T16:22:53.009574Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft on r3: update ~970/31,281 at 16.4 s/update. Ledger reserve is now proportional (fc8a892a). Review workstreams running: e2e progression for both lines (lockstep), task-success gate (e7f09db8), nested+QAT, efficiency, revival (harness/memetic/RL/llama.cpp), housekeeping. One combined v4 handoff is pending those. Memory beside v4 is tight (about 5–8 GB headroom), so small jobs queue.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `natlang-groq-oss120b-musique-equivalence-v1` |  |  |  |
| `natlang-luna-clinc-v17-lane06-v3` | low | 1.5 / 0.2 |  |
| `natlang-luna-e2e-musique-answer-s52002-lane07` | low | 1.0 / 0.2 |  |
| `natlang-luna-e2e-musique-s52001-lane01` | low | 1.5 / 0.5 |  |
| `natlang-luna-e2e-musique-s52001-lane02` | low | 1.5 / 0.3 |  |
| `natlang-luna-e2e-musique-s52002-lane03-v3` | low | 1.0 / 0.3 |  |
| `natlang-luna-e2e-musique-s52003-lane05-v2` | low | 1.0 / 0.7 |  |
| `natlang-maple-dgx-tunnel-20261005` |  |  |  |
| `natlang-nvidia-clinc-review-v2` |  |  |  |
| `natlang-pop-muon-main-v3-resume-v15` | high | 5.5 / 9.2 | neuralese-direct-lfm-muon-main-20261012-v3-pop-launch.json |

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

- 2026-10-12T17:56 dgx-claude-d8484b5f: Owner clarification: program-header is an efficiency suggestion, not a rule
- 2026-10-12T18:28 dgx-claude-d8484b5f: Owner: e2e data includes everything
- 2026-10-12T18:33 dgx-claude-d8484b5f: Lockstep: take the windowing change in your LFM handoff
- 2026-10-12T19:19 dgx-claude-d8484b5f: Owner: projection and backbone train jointly; frozen TCP is diagnostic only
- 2026-10-12T19:19 dgx-claude-coreweave: CoreWeave sandboxes: plan + controller (b00b2c94); Dockerfile deps now requirements.txt
- 2026-10-12T20:18 dgx-claude-d8484b5f: Owner: CoreWeave GPU sandboxes for training and teacher generation
