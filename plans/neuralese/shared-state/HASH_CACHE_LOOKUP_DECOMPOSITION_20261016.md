# Incremental lookup of verified file hashes

The supervisor observed a 40 MB append-only hash log being parsed from its
beginning for every small artifact lookup. This delayed a focused check for
minutes. Keep the existing file identity and digest semantics; improve only the
lookup mechanism shared by all models and data jobs.

1. Read complete existing log entries once into a process-local index.
2. On an append, read only complete new entries. Keep the original first-entry
   lookup semantics and leave an incomplete final line for the next append.
3. Rebuild after cache replacement, truncation or a same-size rewrite. Switching
   the configured cache path replaces the process-local index.
4. A miss still uses the existing archive receipt or exact file-byte hashing and
   appends through the existing writer. No model identity or source digest changes.

The index and lock are crisp lookup plumbing. They make no admission or training
policy decision. The persistent JSONL remains the existing shared cache format.
