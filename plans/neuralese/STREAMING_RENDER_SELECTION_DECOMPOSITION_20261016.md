# Compress selection while it is written

Search/OWL's 6.9 GB of compressed source expands into far more scratch than
available disk. The prior selection path wrote all selected shards as plain
JSONL, then compressed them only after the entire input finished. It consumed
51,337,396,628 bytes before selection was safely stopped and the incomplete
scratch removed; immutable sources and protection proofs were retained.

This is exact storage plumbing, not model-facing policy. Selection, split
checks, omissions, trajectory sharding and record bytes remain unchanged.
When `--compress-text` is set, selected shards use the existing shared streaming
JSONL writer from their first byte. ExitStack closes all compressors and checks
their exits before selection state becomes resumable. An interrupted selection
has no completed state and the existing restart cleanup removes only scratch.
Old completed plain selection remains readable and can still be compressed by
the existing post-selection compatibility path. No uncompressed corpus-sized
intermediate is needed. Compression workers belong to the rendering ledger unit;
admit memory for all shard writers and measure its actual peak.

Ledger `streaming-render-selection-20261016`: 24 tests passed, including exact
decoded shard byte/metadata parity (Unicode and held split aliases), unchanged
duplicate/chain handling, and failure propagation before resumable state when a
writer fails to close. `streaming-render-selection-quick-20261016` passed quick
checks. The production restart must measure all 128 writers under a 32 GB
initial ledger claim; later phases still use two render workers.
