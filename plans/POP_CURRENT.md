# POP current state

Generated 2026-10-12 19:02 UTC by `python3 scripts/current_state.py --machine pop`. Do not edit by hand: regenerate.
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

Updated 2026-10-12T19:01:56.191952Z by pop-codex-01a0e480.

Pop neuralese 2026-10-12 21:02 CEST: LFM direct Muon main checkpoint remains full update11204; high-priority unit natlang-pop-muon-main-v3-resume-v13.service active, doing CPU data-only plan over r3 then expected GPU continuation with Q4/family/task cadence. v11 reached 5.5GiB before first update because prepared token windows stayed resident; stopped unchanged. Shared fix 160829d3 file-backs 32MiB token/role shards (unlinked mmap), focused tests 8/8 and quick check 16/16. v12 failed preflight on stale operational package hashes; v13 recipe re-pinned, live. E2E 200-update arm checkpoint preserved but unqualified: held written CE .471→3.706, crisp .364→2.944. All 200 training records were one family (decision_extract_chain) due ordered one-cohort cursor; DGX owns seeded stratified sampler and TCP LR/freeze A/B, per-family CE. Pop rendering/masking/eval-load trace found same core path, training also has context CE1 and feedback .25. Three Luna CLINC workers active (lanes03-05); lanes01/02 and previous lane06 finished. Two replacement slots should move to E2E multi-call v29 worlds after new reviewed queues. QAD full-precision base + matching head anchor pinned; fresh QAD gate 18/48; this continuation retains original LFM weights. Disk root ~45GiB, storage ~18GiB.

## DGX status page

Updated 2026-10-12T16:22:53.009574Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft on r3: update ~970/31,281 at 16.4 s/update. Ledger reserve is now proportional (fc8a892a). Review workstreams running: e2e progression for both lines (lockstep), task-success gate (e7f09db8), nested+QAT, efficiency, revival (harness/memetic/RL/llama.cpp), housekeeping. One combined v4 handoff is pending those. Memory beside v4 is tight (about 5–8 GB headroom), so small jobs queue.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `natlang-luna-clinc-v16-lane03-v3` |  |  |  |
| `natlang-luna-clinc-v16-lane04-v3` |  |  |  |
| `natlang-luna-clinc-v16-lane05-v2` |  |  |  |
| `natlang-maple-dgx-tunnel-20261005` |  |  |  |
| `natlang-pop-muon-main-v3-resume-v13` | high | 5.5 / 1.7 | neuralese-direct-lfm-muon-main-20261012-v3-pop-launch.json |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | Fresh LFM2.5-350M direct v3 lineages start from the pinned full-precision LiquidAI QAD source and matching clean heads. |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (22): `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-muon-main-20261012-v3-pop-launch`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u10006-pop-launch`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`

## Corpus registry

1371 live entries in `training/neuralese_corpora.json` (114 dgx, 1257 pop); 95 carry training admission.

## Open coordination requests to pop

- `2026-10-11T09:29:18.442129Z-pop-aab3`: V41 enum diagnostic and V40 no-enum bundle ready
- `2026-10-12T16:17:43.543343Z-dgx-f296`: Gate is report-only; task-success eval in the cadence; LFM base 0/48 root cause (1cc33160, e7f09db8)
- `2026-10-12T16:24:16.095493Z-dgx-345c`: Shared e2e lineage on main (983b9fc6): LFM qualification arm launch + handoff path of your LFM main into the shared lineage
- `2026-10-12T17:07:37.233564Z-dgx-1fd4`: Held-task base 0/48: 27 model_incomplete come from scoring the base under program-header rendering (no tool schemas)
- `2026-10-12T17:49:30.017854Z-dgx-e93c`: LFM main handoff into the shared lineage: recipe ready (neuralese-direct-lfm-muon-main-20261012-v3)

## Latest decisions (coord)

- 2026-10-12T17:07 dgx-housekeeping: One recipe lineage: 104 discarded recipes deleted on main (9513c792); delete your copies and run dirs
- 2026-10-12T17:48 dgx-eval: Task gate: LFM base reference on packet v2 = 1/16; eligibility fix; parity (de0d8d35, 4574dbb6, 93d76ad2)
- 2026-10-12T17:55 dgx-claude-d8484b5f: Owner: do whatever is necessary to make e2e work well; proceed with your explicit transition
- 2026-10-12T17:56 dgx-claude-d8484b5f: Owner clarification: program-header is an efficiency suggestion, not a rule
- 2026-10-12T18:28 dgx-claude-d8484b5f: Owner: e2e data includes everything
- 2026-10-12T18:33 dgx-claude-d8484b5f: Lockstep: take the windowing change in your LFM handoff
