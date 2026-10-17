# Exact teacher graph content-index reuse

Policy: preserve every usable captured teacher action, omit only exact duplicate token/weight graph content, and keep the earliest source-file/row identity. Tokenizer-specific model views remain separate. Cache availability is neither admission nor training exposure.

1. Reuse the existing common row-index cache for the pure result of validating one prior graph file and computing its content fingerprints.
2. Bind the exact input SHA, expanded tokenizer fingerprint, converter source, existing seven-file source/parser policy and graph facade. Changed bytes or policy require the original complete parse.
3. Accept only a correctly typed cache key, nonnegative row count, lowercase content hashes, nonempty first-row identities and an exact metadata checksum. Missing, unreadable or invalid entries fall back to parsing; train-split and tokenizer validation remain mandatory on a miss.
4. Recheck the input SHA before publishing or consuming the result. Cache writes use the existing atomic row-index helper and may fail without failing conversion.
5. Merge each file's first-row fingerprints in the original input order with setdefault. Return the same source bindings and row counts. Source, protection, actual decision materialization, rendering, graph admission and per-model outputs retain their existing checks.

This is crisp identity, validation and caching plumbing for an unchanged natural-language data policy. It changes no numerical runtime, seven-file source policy, generation prompt or training state.
