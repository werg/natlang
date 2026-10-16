# Offline preference source-index continuation

## Policy

For a fixed normalized preference export, current source-group protection, and
current decision holds, compute source eligibility once and reuse it across the
LF/Mellum tokenizer views. Reuse is allowed only when every byte-bearing input
and the implementation that interprets it match. A completed source index
accounts for every normalized source row as either an eligible complete pair or
an explicit omission. Partial work is recoverable only through its exact
identity and a durable record frontier. Tokenizer-specific rendering,
objective checks, graph/source joins, and cross-model split/held checks remain
per-view and must still run.

## Functions and exact mechanism

1. `bind_index_inputs`: bind the normalized source, export receipt, global
   source-closure receipt, current protection proof, exact decision-hold set,
   and parser/eligibility implementation hashes. This is crisp because these
   are exact file identities.
2. `scan_source_eligibility`: validate original source-file/revision fields,
   stream the existing adjacent-pair reader, and write one compact row per
   pair/omission. The source row order and pair indices are retained. This is a
   crisp verifier over immutable source records; it does not choose or rewrite
   supervision.
3. `commit_index_batches`: after each bounded batch, fsync appended JSONL rows
   and atomically publish the matching frontier and byte offset. On recovery,
   truncate only bytes after the committed offset, verify every committed row,
   then resume the same source scan from its recorded unit frontier. A final
   receipt binds total source rows, units, index bytes, and SHA-256.
4. `reuse_source_index`: derive the compact expected-membership and omission
   maps from the complete verified index. The two model tokenizers share this
   source accounting but not rendering results.
5. Existing per-view verification continues to rerender both original
   alternatives with the actual tokenizer, compare exact graph/objective
   hashes, and validate current protection, split, held, output, and source
   joins before writing an admission proposal.

## Boundaries

The index is an optimization and recovery witness, not admission by itself. It
does not authorize a training claim, alter source labels, change pair masks, or
replace actual model-specific render verification. Any changed source,
protection, hold, reader, or eligibility implementation selects a different
index identity and requires a fresh scan. Output publication remains under the
existing admission output lease.
