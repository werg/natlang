# Native state serving decomposition (2026-10-15)

## Policy and exact mechanisms

Serving treats a native state block as immutable model state, not as projected Neuralese vectors. Natural-language policy chooses whether a consumer imports exact native state or takes the explicit foreign-text fallback; it rejects a block whose provenance or architecture cannot support the requested operation. Exact code is limited to content identity, tensor layout checks, atomic persistence, HTTP transport, tokenizer/model binding, and the splice/capture calls that cross the model-runtime boundary.

1. **Create a producer span** — the shared model execution owns the decision and capture semantics in `model/shared_state.py`; serving dispatch is plumbing. Capture the token span between the dedicated produce markers, including text, token IDs, positions, embedding rows, and every declared layer state. The returned NativeStateBlock is checked against the loaded model's checkpoint, tokenizer, embedding, architecture, and layer identities.
2. **Persist and transport it** — the store owns exact immutable durability: canonical metadata, named tensors, tensor dtypes/shapes/bytes, and the format version determine a content ID. The existing store's holds, pins, collection, disk spilling, and owner scoping remain the single lifecycle mechanism. HTTP is an outside-world adapter; TypeScript uses the same canonical identity and wire fields.
3. **Resolve a consumer reference** — message parsing recognizes a distinct native-state part and carries its ID as structured input; it never serializes raw tensor state into prompt text. Exact-compatible blocks invoke the shared model's splice operation at the consumer marker and current positional boundary. A foreign block can use its text only through explicit target tokenization and construction of target-native cache state; raw foreign KV is never passed as compatible.
4. **Return producer references** — responses preserve the text and return a distinct native-state content part plus verified metadata. Structured references participate in existing pin/collect behavior exactly like other content-addressed blocks.
5. **Compatibility and cutover** — the legacy `nz1_` projected-block route remains available for diagnostic parents. The new format uses a disjoint versioned identifier and rejects cross-version interpretation. Legacy removal is a later qualified cutover decision after both model serving and transport consumers use this path.
6. **Capture a replayable native sampling turn** — when requested, the serving boundary emits `natlang.native-rl-sampling/1`: exact policy identities; ordered prompt token/native-state segments; every sampled token, its normalized log probability, the role used on its next model input, and terminal status; produced state IDs with half-open ranges; and the effective sampling seed/law. The runtime keeps this record on the raw model response through collection and rollout emission. Native state IDs are transitive artifact references: the collection actor archives each referenced artifact before it releases its server-side owner.

## Components

| Component | Natlang treatment | Reason |
| --- | --- | --- |
| Exact-vs-best-effort native import policy | Natural-language serving policy, with a single pluggable interface where it must be selectable | It decides compatibility and fallback behavior. |
| Producer/consumer marker semantics and model-state capture/splice | Shared natural-language model functions plus crisp tensor boundary helpers | These are model behavior; tensor slicing and cache insertion require exact code. |
| NativeStateBlock validation and content identity | Crisp verifier | Byte-for-byte identity and shape/dtype checks must be deterministic across Python and TypeScript. |
| Persistence, pin/hold/collect, HTTP and client transport | Existing crisp store/service mechanisms, extended for a second versioned record kind | Atomic writes and external transport are exact durability/service boundaries. |
| Reference rendering and response parsing | Natural-language message policy with crisp structured-part parsing | References remain values; raw state is never rendered as ordinary prompt text. |
| Native rollout sampling receipt | Exact serving/collector metadata beside the natural-language task trajectory | Token IDs, sampled log probabilities, marker roles, range boundaries and artifact identities must round-trip unchanged for the native graph objective. |

## Required verification

Cross-language fixtures must show identical IDs and bytes for the same metadata/tensors, reject corruption and shape/dtype disagreement, preserve ownership and restart durability, and round-trip structured references. Serving tests must exercise an actual producer capture and exact consumer splice through the shared model path, and prove the foreign fallback rebuilds target-native state from text without reusing foreign KV. Model-facing marker/instruction changes are recorded for live measurement before adoption; no new model-facing sentences are required merely to add the transport representation.
