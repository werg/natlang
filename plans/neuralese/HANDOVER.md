# DGX current state

Generated 2026-10-12 17:07 UTC by `python3 scripts/current_state.py --machine dgx`. Do not edit by hand: regenerate.
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

Updated 2026-10-12T16:22:53.009574Z by dgx-claude-d8484b5f.

Mellum v4 direct_sft on r3: update ~970/31,281 at 16.4 s/update. Ledger reserve is now proportional (fc8a892a). Review workstreams running: e2e progression for both lines (lockstep), task-success gate (e7f09db8), nested+QAT, efficiency, revival (harness/memetic/RL/llama.cpp), housekeeping. One combined v4 handoff is pending those. Memory beside v4 is tight (about 5–8 GB headroom), so small jobs queue.

## POP status page

Updated 2026-10-12T16:50:44.128718Z by pop-codex-01a0e480.

Pop status 2026-10-12 18:50 CEST
- Main LFM2.5-350M MuonSR direct SFT saved update 10006, restarting via natlang-pop-muon-main-v2-resume-fp8-v8.service from full checkpoint. Previous step9000 strict direct export remains saved but its external launch receipt hash was invalidated when launcher finalized mutable receipt at stop.
- Fixed strict exporter locally to freeze an immutable byte snapshot of launch receipt. E2E 200-update mechanics arm blocked before training by that provenance issue; will re-export from current checkpoint and qualify after code publication.
- Six Luna workers running: V14 lanes01-03, V15 lane6, V16 lanes01-02. V16 lanes03/04 reviewed and sealed as replacements at cap6.
- Gemini PubMedQA 43/43 successful after seven-case retry; strict static adapter produced 32 eligible labels / 4 cases / 54 turns, held pending admission. Resume and subset bugs fixed locally.
- Protected CPU candidate task packet paused after 12 complete cases to free host RAM; case12 partial must retry. DGX owns held packet v2.
- QAD LFM2.5-350M under consideration as quantized deployment and future base candidate; current lineage unchanged pending matched Natlang benchmark.

## Running on this machine

| Unit | Priority | Budget / used GB | Recipe |
| --- | --- | --- | --- |
| `eff-v4-windowing-plan` | medium | 10.0 / 0.5 |  |
| `heldv2-base-lfm-repeat` | medium | 5.0 / 2.7 |  |
| `memetic-author-20261012` | medium | 5.0 / 3.1 |  |
| `natlang-full-check-e7f09db8b1` | medium | 13.5 / 0.7 |  |
| `natlang-memory-guard` |  |  |  |
| `natlang-neuralese-direct-mellum-v4-161355` | high | 101.8 / 68.8 |  |

## Declared lineages (training/neuralese/line-parity.json: next launch per line)

| Stage | Line | Recipe | Extends | Description |
| --- | --- | --- | --- | --- |
| direct | lfm | `neuralese-direct-lfm-v3` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on base LFM2.5-350M (LiquidAI/LFM2.5-350M, revision 9e6c6ccf; he… |
| direct | mellum | `neuralese-direct-mellum-v4` | neuralese-direct-v3 → neuralese-direct-v2 → neuralese-direct-v1 | neuralese-direct-v3 (union r2, three phases: full-prompt warm-up 0.5 epoch, program-header direct_sft 1 epoch, whole-trajectory direct_trajectory 1 epoch) on BF16 Mellum2.1 with raw-token-v1 heads. |
| e2e | lfm | `neuralese-e2e-lfm-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on LFM2.5-350M (Pop). |
| e2e | mellum | `neuralese-e2e-mellum-v1` | neuralese-e2e-v1 | neuralese-e2e-v1 on Mellum2.1-12B-A2.5B (DGX). |

Recipes on disk (20): `luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3`, `luna-neuralese-direct-lfm-base-20261011-v8`, `neuralese-direct-lfm-muon-main-20261012-v1-full`, `neuralese-direct-lfm-muon-main-20261012-v2`, `neuralese-direct-lfm-muon-main-20261012-v3`, `neuralese-direct-lfm-v1`, `neuralese-direct-lfm-v2`, `neuralese-direct-lfm-v3`, `neuralese-direct-mellum-v1`, `neuralese-direct-mellum-v2`, `neuralese-direct-mellum-v3`, `neuralese-direct-mellum-v4`, `neuralese-direct-mellum-v4-from-v3-warmup`, `neuralese-direct-v1`, `neuralese-direct-v2`, `neuralese-direct-v3`, `neuralese-e2e-lfm-v1`, `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`, `neuralese-e2e-mellum-v1`, `neuralese-e2e-v1`

## Corpus registry

1369 live entries in `training/neuralese_corpora.json` (112 dgx, 1257 pop); 93 carry training admission.

## Open coordination requests to dgx

- `2026-10-12T00:09:18.498511Z-pop-1183`: Coordinate neuralese-learning test split before fixture work
- `2026-10-12T13:57:24.496893Z-pop-0adb`: Need bounded r3 missing-doc source IDs
- `2026-10-12T16:49:51.773635Z-dgx-5db1`: e2e QAT/family treatment: confirm before I push (shared q4 in neuralese-e2e-v1)
- `2026-10-12T16:52:14.792718Z-dgx-e1f6`: dgx-nested -> dgx-e2e: e2e QAT now in the shared base (1f1042c3); please check the LFM e2e trainer path

## Latest decisions (coord)

- 2026-10-12T16:08 pop-luna-salvage-static: Pop Luna generation cap raised to six
- 2026-10-12T16:17 dgx-eval: v4 handoff now also takes task_eval; behaviour gate report-only (e7f09db8)
- 2026-10-12T16:18 dgx-nested: Shared base neuralese-direct-v3: light QAT (q4) + nested family for LFM; objective continuation handoff (7bbfd00f, f57eb9b6)
- 2026-10-12T16:54 pop-codex-01a0e480: Owner: fresh LFM restart from QAD source weights
- 2026-10-12T17:03 dgx-eff: Shared base neuralese-direct-v3: whole-window train supervision + merged decision chains (5a5d8e69)
- 2026-10-12T17:07 dgx-housekeeping: One recipe lineage: 104 discarded recipes deleted on main (9513c792); delete your copies and run dirs
