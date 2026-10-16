# Native RL with distinct invocation actors

Policy: collect actual per-call adapted behavior, retain its immutable sampling
probabilities and exact artifacts, then improve declared current actor
coefficients through the shared one-pass invocation graph. A common base is a
base identity, never a claim that adapted calls sampled from the unadapted
model. The behavior graph has a versioned policy descriptor and its own SHA;
it has no invented single effective model identity.

The typed natural-language program chooses adapter scopes and communication
roles. Crisp services authenticate the scopes, preserve artifacts, execute the
shared graph/objective and publish exact optimizer continuations. There is no
new actor-selection policy in crisp code.

1. Derive each behavior invocation's exact actor from the host sampling receipt.
   Require sampler/3 for adapted calls, validate its ordered coefficient/hex-scale
   binding and exact common base, and authenticate written raw/compact state and
   consumer receipts against their actual producing/consuming actors. Existing
   uniform sampler/1 and /2 behavior stays under its original contract.
2. Compile the actual call graph once using the existing native replay compiler.
   Different actors may share the exact immutable base, never merely dimensions.
   Internal imports alias the genuine producing occurrence; external observations
   keep actual immutable raw state. No text fallback may impersonate a KV edge.
3. Bind a versioned behavior policy descriptor to the common base and exact
   per-episode/per-invocation actors. The existing durable block-bundle policy_id
   is an opaque ownership ID; use the descriptor SHA for this mode, retaining all
   coefficient/basis and raw producer/consumer dependencies through the existing
   bundle walker. Do not label that SHA a model checkpoint. Preserve original
   collection and source provenance separately from the new behavior binding.
4. Admit a versioned native-policy objective that explicitly separates common
   base, immutable behavior descriptor and current invocation actor receipt.
   Existing uniform objective/1 remains unchanged. Every current actor must
   retain its behavior actor's exact base, ordered scales and frozen bases;
   coefficient values/IDs may evolve. No behavior logp, reward or source row is
   rewritten by subsequent optimizer steps.
5. Use the existing graph packer ownership/name maps to bind current actors to
   every actual packed invocation. Execute the existing layer-major actor bank,
   sparse attention/hybrid handling and shared GRPO/RFT objective. Forward,
   checkpoint recomputation and backward share one live operation lease.
   Consumer loss reaches current producer coefficients through internal imports.
6. Revalidate each external state's actual source and each behavior/current
   consumer. A changed consumer requires explicit same-base state transport;
   strict default behavior cannot reuse old external KV silently. Preserve exact
   source/target receipts in artifacts and replay. Compact learned adaptation
   retains its existing exact adapter/model contract and cannot masquerade as
   raw exact-base transport.
7. Publish immutable coefficient updates with full per-invocation continuation,
   exact named moments and archived parents. Reopen stores and restore complete
   dependencies between updates. Reject unknown actors, changed bases/basis/
   scales, forged source bindings, held episodes and stale descriptor digests.
8. Validate both real ports using actual adapted Engine generation/capture,
   recorded behavior-logp replay, strict incremental current-policy gradients,
   two durable GRPO updates, producer-gradient reachability, cached convolution,
   actual FP8 projections where claimed, and unchanged unadapted state. Keep
   public capability unadvertised until independent review and required target
   runtime checks. No full-size or cloud claim follows from tiny local fixtures.

Implementation is pending. Existing policy/1 rejects heterogeneous actors; that
honest rejection stays until this entire admission/archive/current-vs-behavior
contract is implemented. Graph-CE support alone is not RL completion.
