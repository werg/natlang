# Full-check temporary storage

The full check encountered actual ENOSPC while creating runtime fixtures.
Keep its verdict red until the affected suite reruns successfully. This change
is exact test-process and filesystem plumbing, with no model-facing policy.

1. Give each checked commit its own temporary directory. Select its parent
   through `NATLANG_FULL_CHECK_TMPDIR`, allowing the DGX service to use the
   appropriate storage without changing other processes' temporary directories.
2. Pass that exact directory through the memory ledger into every check child.
3. Mark directory ownership by the complete commit SHA. Clean only marked,
   nonsymlink directories belonging to a terminal check; keep the durable log.
4. Existing admission, test commands and complete-verdict requirements remain.

Storage diagnosis, 2026-10-16: the external HDD caused real NVMe guard failures,
SQLite lock contention and test timeouts. The same old-worktree affected tests
passed unchanged on SSD (3 recipe, 37 call-store, 57 evaluation/collector cases).
The standing service now selects `/home/werg/data/full-check/tmp`; retain the
old full-check failure and obtain a new complete verdict.

Frozen test fixtures can remove directory write permission. After validating the
terminal run marker, restore owner read/write/search only on owned directories
inside that run; never follow symlinks or modify their targets. Remove children
first and the marker last, so an interrupted cleanup keeps its ownership proof.
Regression checks cover frozen nested directories, untouched external symlink
targets and marker retention after cleanup failure. This remains crisp cleanup
mechanism, not test-selection policy or an assertion bypass.
