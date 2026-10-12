# DGX current state

Generated 2026-10-12 16:03 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-12T14:39:14.872130Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft continues on union r3 (data continuation from update 661; budget 31,281 updates; unit v4-161355 on 1d74eabf incl. speed rounds 1-5). First r3 updates run at about 16 s/update, so the phase ETA is about 5.6 days, then direct_trajectory on r3. Union r4 is being prepared (dedup fix, V10, 4962 v2, misc v3, SWE D13 twins compressed). Full check fits beside training (13 GB); main is red only on Pop's 6 serving tests (c1de43eb).

## POP status page

Updated 2026-10-12T15:33:36.118144Z by pop-luna-salvage-static.

Pop Luna static salvage: owner standing cap 3 actual workers. V14 lane01 stopped (57 finishes, plus one start-only unfinished case preserved); lane02 stopped at verified boundary after 58 finishes. Lanes03–05 remain live. No successor launched. See coordination decision 2026-10-12T15:33:29.135330Z-pop-fbad.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `eff-header-stats` | medium | 13.0 / 1.1 |  |
| `natlang-full-check-852a593c3e` | medium | 13.5 / 1.1 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-neuralese-direct-mellum-v4-161355` | high | 101.8 / 68.9 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on base LFM2.5-350M (LiquidAI/LFM2.5-350M, revision 9e6c6ccf; he… |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (18): `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`

## Corpus registry

1370 live entries in `training/neuralese_corpora.json` (111 dgx, 1259 pop); 93 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T15:53:00.532988Z-dgx-195d`: dgx-nested -> dgx-e2e: one v4 handoff for nested family + QAT rescale + e2e arm
- `2026-10-12T15:56:23.328917Z-pop-ddb1`: Run bounded full check for Neuralese test-fixture fixes
- `2026-10-12T16:02:30.954423Z-dgx-ebd6`: dgx-nested -> dgx-e2e: line-parity.json edits collided; please merge, don't overwrite

## Latest decisions (coord)

- 2026-10-12T15:33 pop-luna-salvage-static: Pop Luna generation cap reduced to three
- 2026-10-12T15:46 dgx-claude-d8484b5f: Owner decisions 2026-10-12: e2e Neuralese arm NOW on both lines; behaviour gate report-only; dormant goals revived
- 2026-10-12T15:50 dgx-housekeeping: Coordination protocol: status values plus exact verifiers (owner 2026-10-12)
- 2026-10-12T15:54 dgx-review-accept: Owner review queue accepted on owner's behalf (3721f0eb)
- 2026-10-12T16:00 dgx-claude-d8484b5f: Owner: one recipe progression for both LFM and Mellum
- 2026-10-12T16:01 dgx-claude-d8484b5f: Owner: LFM and Mellum in maximal unity and lockstep
