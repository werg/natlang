# Native-state text capture decomposition (2026-10-15)

The natural-language caller decides when supplied text is a body/value and what typed slot receives it. The native
service performs only exact model-facing tokenization, causal execution, state capture, and durable storage. It does
not decide that a new text value should be created, claim that a native state reproduces ordinary-text logits, or
use the old projected-value read/write transport.

1. `native.text-write` receives exact source text and its declared type. The HTTP boundary validates the request and
   calls the shared Engine operation. Crisp code is needed for an external service boundary and an exact receipt.
2. The Engine escapes control-token spellings as literal source text, tokenizes with the checkpoint tokenizer, and
   executes `produce-open`, the payload with the learned producer transform enabled, and `produce-close` through
   `native_step`. It stores only the payload KV/hybrid state in `NativeStateBlock`; the source receipt retains all
   actual token IDs and roles, including both delimiters. Crisp code is required to reproduce the exact causal
   sequence and state tensors.
3. `native.init-body` replaces exactly one declared native placeholder with the supplied text in the rendered call.
   It executes the entire preceding context, imported native states, producer delimiters, payload, and following
   context in source order, capturing only the payload span. If tokenizer merges prevent exact span isolation, it
   returns a structured boundary error instead of fabricating body coordinates. Imported native states remain
   separately identified segments; cross-model text fallback is recorded as ordinary target-token input.
4. The shared native-state store commits the captured immutable tensors and provenance under the content ID and
   owner hold before the endpoint returns. `source_capture` binds the policy identity, ordered source segments,
   payload half-open own-token bounds, and SHA-256 of exact source text. This is exact durability/provenance plumbing.
5. The old `embed` and projected `init_body` routes remain unavailable on a native checkpoint. `/info` advertises
   the two native capabilities only when these implementations are active. Native `view`, vector combinators,
   and gradient clients remain separate migration work and are not inferred from text capture support.

The response marks `exact:false` for contextual initialization: capturing the actual producer state is exact, but
the imported state plus consume markers is not claimed to reproduce the original plain-text call's logits. Any
future model-facing prompt or error change must be measured on the live executor before adoption.
