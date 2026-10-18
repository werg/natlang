# Declared callback argument boundary

The source contract is the policy: a checked callable with named formal
parameters exposes those arguments to its natural-language child. Ordinary
JavaScript callback extras do not enlarge that contract. An untyped open
natural-language callable still receives every value supplied at its call.

1. The compiler records whether the parameter list came from a full explicit
   callable type or a checked contextual callback signature. Result-only
   annotations and inferred/open calls do not assert a declared boundary.
2. Lowering preserves that fact in the existing callable definition and frozen
   adaptation contract. The lowering protocol changes so cached older output
   cannot be silently interpreted as the new contract.
3. The shared invocation kernel projects excess positional values before tier
   dispatch, call recording, scoped-handle acquisition or model rendering.
   It preserves every declared parameter, including an explicitly declared
   callback index/array; too few required values retain existing checks.
4. Open and legacy unmarked callables keep their existing argument behavior.
   JavaScript rest forwarding remains ordinary JavaScript; this change does not
   reinterpret an unsupported explicit NL rest signature.

This is crisp verification and projection of source-authenticated structure,
not semantic classification of model input. It changes model-visible input
boundaries, so both authoring skills and the live-measurement debt are updated.
No active collector, actor, checkpoint or frozen task runtime is relabeled.
