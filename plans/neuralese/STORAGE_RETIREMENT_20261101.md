# Storage retirement, 2026-11-01

Owner-authorized Megatoken deletion is complete. `/home/werg/megatoken-runs`
is absent; its deletion receipt records 95,022,338,048 bytes (88.5 GiB) reclaimed.

The historical DGX recovery workspace was relocated to
`/mnt/external/natlang-recovery-archive/claude-recovery-20261028` after transfer
verification. Its former internal path is a symlink. Unique edits and recovery
refs remain; clean integrated worktrees, redundant unpacked staging and derived
caches were removed. See the earlier relocation and cleanup receipts.

A subsequent internal-space drop came from new full training checkpoints and
reservations. This follow-up removed two unused single-link files after checking
live process descriptors and mappings:

- The obsolete intermediate checkpoint in rejected-lineage
  `joint-v4-full-successor-20261031/run-20-attempt-5` (15,267,278,848 allocated bytes).
  Final rejected checkpoint72 remains for diagnostics; current production parents,
  good SD52 and independent recovery copies remain protected.
- The disposable materialized `debug/gpu-window-1/inputs/rl-native-objective.json`
  (2,790,764,544 allocated bytes). The completed result and source data remain.

Internal available space rose from 26,301,833,216 to 44,360,671,232 bytes at the
recorded observation, restoring headroom above the 40 GiB floor. Active rolling
checkpoint reservations and pending exports were untouched. Subsequent training
writes change free space; full-state export pins must remain bounded rather than
accumulating one unique full checkpoint at every update.

Receipts are under `/home/werg/data/lfm3-rl-dgx-20261029/ops/`:
`megatoken-deletion-20261031.json`,
`recovery-workspace-relocation-v2-20261031.json`, and
`obsolete-derived-retirement-1101.json`.
