# Search/OWL native-render selection checkpoints

## Policy and exact mechanism

Selection policy remains the existing `render_native_turn_corpus.select` decision order: source order, explicit admission, duplicate handling, source and protected-group checks, then stable shard assignment. This work adds recoverability only; it does not change which rows qualify or their order.

1. Bind each run to the ordered source SHA/size list, selector options, and hashes of every include/exclude/protection input. A checkpoint from a different input or selector implementation is rejected.
2. Process one complete source shard at a time. The Search/OWL input is already divided into 164 immutable compressed shards, so a checkpoint frontier at each source boundary bounds lost work to at most the currently processed shard.
3. Write selected rows to per-output-shard zstd streams. At a completed source boundary, close each touched stream as a complete concatenated zstd frame, flush and fsync its output, and record the frame offset, byte count, and SHA. Then atomically persist the selector's cumulative counters, omissions, blocked chains, dedup IDs, group sets, protected cache, and next source index.
4. On restart, verify the checkpoint binding and every committed frame SHA, truncate bytes beyond the last committed frame boundary, restore the cumulative selector state, and resume at the next source. If interrupted inside a source, its uncommitted compressed tail is discarded and that source is replayed; a committed source is never duplicated or silently skipped.
5. After selection finishes, the existing renderer consumes the concatenated zstd frames through the shared `open_jsonl_text` reader. The final receipt retains the same row counts, omissions, source hashes, and selection semantics as an uninterrupted run.

The policy choice remains in the existing natural-language-decomposed renderer. The new crisp code is limited to content hashing, frame durability, atomic checkpoint replacement, and exact resume validation: these are storage and restart guarantees that cannot be delegated to a probabilistic policy stage.

## Failure and resume checks

- Interrupt during a source after several selected writes, restart, and prove that the incomplete source is replayed once with no duplicate IDs.
- Interrupt after a committed frame and state, restart, and prove the earlier compressed prefix and source frontier remain unchanged.
- Corrupt a committed frame, alter a semantic input binding, or remove a committed shard; resume must fail closed.
- Append junk after the last committed frame; resume must truncate only that uncommitted tail and reproduce the uninterrupted selected streams byte-for-byte.
- Decode multi-frame selected shards with the existing zstd reader and compare ordered rows, counters, blocked-chain state, omissions, and group sets to an uninterrupted selection fixture.
