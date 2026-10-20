# Teacher content identity excludes unrelated graph loss telemetry

Policy: reuse admitted graph content indexes only when the actual graph validation and content fingerprint law is unchanged. Held loss telemetry is not part of that law.

1. Crisp identity: scope the native_graph facade dependency to its exact parse_graph import authority, using the existing source-policy AST mechanism. Keep schema parser, graph compiler, JSON IO, canonical hashing and fingerprint/cache validation functions bound as before. A changed import/parser/fingerprint still changes identity.
2. Crisp continuation: independently reviewed exact new scoped identity may consume the three existing historical content policies, including the immediately preceding whole-facade law3. Verify source SHA, tokenizer, complete rows/fingerprint metadata digest and old cache key before publishing the new-key copy. Preserve immutable predecessors; never parse old source rows merely because unrelated loss code changed.
3. Normal conversion, materializer, protection, paired-action and admission gates remain unchanged. This does not alter source indexes, graph IDs, targets, weights, exposure, model calls or optimizer state.
