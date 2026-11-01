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

Further verified retirements removed local step54/55 checkpoint aliases only
after fresh full SHA readbacks of the independent Pop copies, and obsolete
SD8/JOLT42 states after verifying the useful SD52 fallback on Pop. Raw rollouts,
logs, task results and the rejected final72 diagnostic remain. The temporary
step58 export pin was released after complete external readback. The unused
step59 pin was subsequently retired after its recovery exercise was abandoned
and live-consumer checks passed; the actual newer recovery bootstrap remains
protected. These operations have separate receipts:
`retire-local54-verified-peer-1101.json`,
`retire-local55-verified-peer-1101.json`,
`retire-superseded-rl-states-1101.json`, and
`release-verified58-pin-1101.json` in the same operations directory.

The internal drive had approximately 99GiB available at04:38 CET. This is an
observation, not a permanent allowance: live full-state reservations and exports
still count against the 40GiB operating floor. A background checkpoint copier
also held an unlinked old full checkpoint open; its owner has now exited. The
shared copy default was corrected from idle I/O class to best-effort class2,
priority7, because idle copies starved during sustained conversion traffic.
Complete independent recovery qualification remains in progress; do not retire
its source pins on the strength of an archive enqueue or partial copy.

Follow-up at 06:03 CET: the older `/mnt/external/megatoken-runs` copy is
also deleted. Only three small run configuration files were retained in the
operations receipts directory; `/mnt/external/megatoken-job-workspaces` was
not treated as disposable run state. Its contents may include unique work.
The external deletion receipt is `external-megatoken-deletion-1101.json`;
the unused step59 pin retirement is `retire-unused59-pin-1101.json`.

The latest filesystem observation is approximately 80 GiB free internally and
346 GiB externally. Concurrent checkpoint and recovery writes change these
numbers. The remaining historical workspace retains unique edits, source data,
Git recovery references and required recovery artifacts; completed duplicate
restore trees, raw archive duplicates, derived caches and integrated worktrees
have already been removed.
