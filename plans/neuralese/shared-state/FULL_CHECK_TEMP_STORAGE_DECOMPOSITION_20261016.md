# Full-check temporary storage

The full check encountered actual ENOSPC while creating runtime fixtures.
Keep its verdict red until the affected suite reruns successfully. This change
is exact test-process and filesystem plumbing, with no model-facing policy.

1. Give each checked commit its own temporary directory. Select its parent
   through `NATLANG_FULL_CHECK_TMPDIR`, allowing the DGX service to use the
   external disk without changing other processes' temporary directories.
2. Pass that exact directory through the memory ledger into every check child.
3. Mark directory ownership by the complete commit SHA. Clean only marked,
   nonsymlink directories belonging to a terminal check; keep the durable log.
4. Existing admission, test commands and complete-verdict requirements remain.
