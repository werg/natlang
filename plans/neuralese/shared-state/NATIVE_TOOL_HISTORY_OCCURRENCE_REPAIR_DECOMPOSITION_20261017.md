# Native tool-history occurrence repair

Policy: preserve the actual sampled episode and require exact producer/consumer occurrences before policy training. The supervisor may repair missing metadata only when the recorded transport arguments, actual server capture, and known writer agree exactly. No resampling, reward change, state substitution, optimizer change, or relaxed verifier is allowed.

1. The runtime's crisp graph recorder decodes JSON tool-call arguments before traversing nested native-state content parts, matching the existing serving renderer. Content text remains text.
2. A crisp historical verifier compares each captured native import with those same recorded tool argument parts and its exact known writer. It adds only missing read events to a new graph; all original events, sampled tokens, targets, policies, rewards, and source bytes remain unchanged.
3. The repair emits a new immutable rollout artifact and a receipt binding original and repaired SHA-256. Existing strict whole-episode verification validates the complete repaired graph. Ambiguous writers, mismatched IDs, source occurrences, or stop-gradient flags fail closed.
4. Operational adoption explicitly binds the repair receipt and artifact. Original sealed request and collector outputs remain preserved. The repair is exact durability/provenance plumbing, not model-facing policy or a new objective.
