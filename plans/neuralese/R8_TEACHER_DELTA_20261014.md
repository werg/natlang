# Protected R8 teacher additions (2026-10-14)

Build a separate immutable delta for the admitted teacher renders omitted from
R8. Do not rewrite R8 or make the cloud PH launch wait for this build. The
delta can enter a declared full-state data continuation after its global
held/source proof and tokenizer-twin check pass.

## Inputs and quality decision

The canonical checkouts have the following four members in all six views
(24 files). Pop verifies immutable manifests before the build:

| Member | Base directory | LFM/Mellum full, program-header, whole-trajectory |
| --- | --- | --- |
| `clef-20261014` | `runs/natlang-native-turns-clef-pools-20261014-v1-render` | `RENDERING/TOKENIZER/text-data.jsonl` |
| `backlog-v5` LFM | `runs/luna-held-shapes-closed-backlog-v5-native-turns-20261014-v1/lfm25-render-v1` | `RENDERING/lfm25/text-data.jsonl` |
| `backlog-v5` Mellum | `runs/luna-held-shapes-closed-backlog-v5-native-turns-20261014-v1/mellum21-render-v1` | `RENDERING/mellum21/text-data.jsonl` |
| `qad-recovery-v3` | `runs/qad-r8-recovery-v3-v2-twins-20261014` | `RENDERING/TOKENIZER/text-data.jsonl` |
| `qad-recovery-v4` | `runs/qad-r8-recovery-v4-twins-20261014` | `RENDERING/TOKENIZER/text-data.jsonl` |

The recovery members use only their semantically reviewed v2/v1 sources,
respectively: v3 has 41 approved turns and v4 has 24. Superseded v1 recovery
outputs and exact-success parents held for poor action shape are excluded.

The backlog receipt has 1,188 full, 23,644 program-header and 12,777
whole-trajectory train documents per tokenizer before global protection.
The Clef cohort has 122 full, 2,530 program-header and 2,378
whole-trajectory documents according to the DGX owner. These are source
renders, not the delta's final admitted counts.

The disputed SST source group
`inline-curriculum:s52040:dataset_workbench:dataset7:a` occurs in 19 backlog
program-header and 16 whole-trajectory rows per tokenizer (none in full).
Use the pinned one-group witness
`training/source-quarantines/s52040-dataset7-a-20261014.jsonl` as
`--quarantine-reference`. Its SHA-256 is
`396ebd767c73771e6b558cdce9ba2365e2c22a78f49f3c8e2fb78366706438bc`.
The witness isolates the disputed group without excluding the other source
groups on those rows unless they independently fail protection. The builder
drops every document sharing this group and records the omission. The
MuSiQue render from the R8 build remains a quarantine reference in all six
views. Keep the same tokenizer revisions and source-review registry as R8.

The backlog-v5 structural review found 614 exact-answer parents whose actions
did not follow the required process (notably missing `iterateOn` or natural
language per-item decisions, and hardcoded ticket filenames). Its pinned
source-group witness is
`training/source-quarantines/backlog-v5-teacher-shape-20261014.jsonl`
(SHA-256 `5d87e164347a44c71fb7b56ab2b2cd289fcc012b7c3bad91c0c6bfa0606602ca`).
It contains 1,136 groups; 25 otherwise unflagged parents share a held group
and are conservatively excluded. The exact audit is in
`plans/neuralese/BACKLOG_V5_SHAPE_REVIEW_20261014.md`. This witness is required
for the delta, even though the source and renders predate the review and still
have `training_admission: true` in the registry.

## Build and verify

Use the shared `scripts/neuralese_data/build_native_turn_union.py` once with:

- `--out-dir runs/natlang-r8-teacher-additions-delta-20261014-v1`
  and `--compress-text`;
- `--tokenizer lfm25=/home/werg/natlang/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f`
  and `--tokenizer mellum21=/home/werg/data/models/mellum21-12b-a2.5b-thinking`;
- 24 `--member NAME:RENDERING:TOKENIZER=PATH` arguments, one for each
  Clef/backlog/recovery file above, with member names `clef-20261014`,
  `backlog-v5`, `qad-recovery-v3` and `qad-recovery-v4` consistent across
  renderings/twins;
- six `--held-reference` arguments for R8's full/program-header/whole-trajectory
  LFM and Mellum renders; wait for the protected R8 whole files to finish;
- eight `--quarantine-reference` arguments: the same six
  `runs/natlang-native-turns-e2e-musique-hops-render-20261013-v1/RENDERING/TOKENIZER/text-data.jsonl`
  files as R8, plus the SST and backlog-v5 structural witnesses above.

Run the builder through the DGX memory ledger at low priority. Verify the
receipt's source hashes, source-review version, protected held groups/texts,
zero train/held source-group overlap, and equal document identities in LFM and
Mellum for each rendering. Inspect `source_quality_quarantine` omissions for
the disputed SST, structurally poor backlog-v5, and MuSiQue derivatives. Keep legitimate rows from any
partly rejected source. Register the final compressed files and proof under
new immutable corpus manifests, sync them to Pop, and add them through an
explicit data continuation after the active PH snapshot. Do not treat the
current registry's `training_admission: true` on the old backlog render as
sufficient to skip this global quality pass.
