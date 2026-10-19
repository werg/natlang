# Exact teacher graph content-index reuse

Policy: preserve every usable captured teacher action, omit only exact duplicate token/weight graph content, and keep the earliest source-file/row identity. Tokenizer-specific model views remain separate. Cache availability is neither admission nor training exposure.

1. Reuse the existing common row-index cache for the pure result of validating one prior graph file and computing its content fingerprints.
2. Bind the exact input SHA, expanded tokenizer fingerprint, the actual fingerprint/validation functions, recursively selected graph-schema parser law, graph-plan/facade, JSON reader and canonical hashing law. Loader classification and unrelated converter admission code do not affect this pure content cache. Changed called semantics require the original complete parse.
3. Accept only a correctly typed cache key, nonnegative row count, lowercase content hashes, nonempty first-row identities and an exact metadata checksum. Missing, unreadable or invalid entries fall back to parsing; train-split and tokenizer validation remain mandatory on a miss.
4. Recheck the input SHA before publishing or consuming the result. Cache writes use the existing atomic row-index helper and may fail without failing conversion.
5. Merge each file's first-row fingerprints in the original input order with setdefault. Return the same source bindings and row counts. Source, protection, actual decision materialization, rendering, graph admission and per-model outputs retain their existing checks.

This is crisp identity, validation and caching plumbing for an unchanged natural-language data policy. It changes no numerical runtime, seven-file source policy, generation prompt or training state.

The reviewed classification refactor preserves the original parser's validation
sequence, graph-plan call and returned fields while extracting `_graph_fields`;
it does not change graph fingerprints or accepted source rows. Under only the
exact reviewed current law identity, reuse the two authenticated historical
content-policy keys after complete source/tokenizer/key/metadata checks. A changed
called parser, fingerprint or hashing law refuses this migration. Physical module
provenance remains separate from this semantic cache identity. This continuation
also covers historical teacher files represented by later filtered execution
views; absence of an exact active-leaf SHA does not invalidate an already valid
content cache. Missing genuinely new content still runs normal validation.
