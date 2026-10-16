# D20 SWE source streaming decomposition (2026-10-16)

## Decision and stages

1. Read the immutable acquisition receipt and source-file inventory. Crisp code verifies file byte lengths and SHA-256 because those are exact filesystem facts.
2. Classify each source path before retaining parsed rows. Only `transcripts/`, `transcripts_subagent/`, and `transcripts_compact/` are candidates for complete conversational conversion. Other supported source tables remain accounted as explicit non-transcript omissions, one row at a time; they are never accumulated as a fake transcript.
3. For an eligible transcript file, retain the complete file's event rows as the current ancestry algorithm requires the full causal graph. Resolve the exact source-index bindings, reconstruct complete contexts, and preserve every original event's row identity in the per-event export accounting. This is existing policy and rendering, not a new task interpretation.
4. The small relational indexes stay bound to the four receipt-listed tables (`sessions`, `session_logs`, `subagent_tasks`, `repositories`) and are verified on each reuse. Their exact join and ambiguity rules remain unchanged; a later disk-backed implementation can replace their storage without changing this interface.
5. Write normalized trajectories and explicit rejects through the existing export writer. No raw source rows, groups, held splits, or causal context are truncated.

## Crisp mechanisms and reason

- File path selection and hash/size checks: crisp, exact source plumbing.
- Ancestry, complete-message reconstruction, repository/session grouping, and quality flags: existing shared adapter semantics; do not change them here.
- Per-file buffering is retained only for eligible transcript inputs because ancestry reconstruction currently needs a whole file. Ineligible multi-gigabyte tables are classified and emitted as streaming row-level rejects before buffering; this addresses the measured D20 OOM without weakening source accounting.
- Export receipts and hashes: crisp, exact durability and provenance.

## Scope and qualification

The measured failure was `conversations.parquet` (6,424,352,043 bytes) and also `commits.parquet` (5,102,596,005 bytes) being buffered before `finish()` rejected their paths. The implementation must prove that these rows are rejected in source order with constant adapter memory and that eligible transcript behavior remains byte-for-byte equivalent on the existing synthetic ancestry fixtures.
