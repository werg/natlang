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

The typed coefficient-policy service implementation is under qualification.
Existing policy/1 keeps its uniform contract. Production native_stage still
rejects adapted actors: executable coefficient-policy collection/admission/
learner scheduling is the next assigned work, not completed by this service.

Concrete contract: keep the existing generic block-bundle/1, whose policy_id is
already an opaque ownership identifier. For invocation policies, that identifier
is the SHA of `natlang.native-invocation-policy/1`: exact common base, explicit
`collector_policy_role: base-runtime-metadata`, and canonical original-episode
SHA → actual invocation actor receipts. Original collection rows are untouched;
their header describes collection runtime, while the new descriptor describes
the actual behavior of every call. Admission verifies both independently and
never treats the base header as evidence of unadapted on-policy samples.
Native-policy-objective/2 stores `base_native_identity` and this immutable
behavior descriptor/SHA; it has no single effective behavior-model identity.
Uniform objective/1 and archive consumers retain their existing contracts.

Implementation evidence (local Torch2.11/cu130, SM121; not target cloud proof):
- Actual distinct-adapter Engine sampling and one-graph behavior replay: both
  ports,2PASS14.92s.
- Actual two GRPO/Adam updates, exact moments and current actor store restart:
  both ports with internal differentiable producer and external explicitly
  imported raw state,4PASS39.52s.
- Independent actual per-token Engine cached current-policy loss and coefficient
  adjoints at BOTH updates, initial sampled logps, unchanged elementwise gates
  and global relative-L2<2e-5:4PASS43.27s.
- Python→TypeScript exact numeric wire plus recursive .nz gradient/optimizer
  closure, immutable behavior dependencies, stale SHA/scales/admission rejection:
  build+4PASS49.51s.
- Full typed HTTP two-update eviction/reopen, raw external restoration and .nz
  fixture: build+4PASS59.32s; absolute-WT quick PASS, unit
  `native-invocation-policy-typed-archive-fix-20261016`.
- Combined uniform-policy regression:13PASS with the single new external typed
  case exposing missing archival, now repaired by the existing shared native
  archival helper and rerun above. Original failed check is not called green.
  Capabilities/default recipes remain unchanged.

Current policy artifacts bind a source-name→actual packed-name bijection. The
optimizer publishes and archives the whole continued actor inventory, including
unrelated current actors; it does not invent a global effective identity.
Gradient archives retain original behavior coefficient/basis dependencies as well
as current actors. Per-external-occurrence checks bind the source, original
consumer, current consumer and actual logical position. Full-size multi-actor
memory/cost and target NGC/H100 numerics remain unqualified.

A live typed fixture found that nativeGraphGradient restored remote raw inputs
without archiving them locally. After remote GC the next update lost the input.
The client now calls the existing archiveNativeState helper for every restored
external input, preserving source coefficient/basis closure and exact raw bytes.
No replacement zero state or reencoded text is permitted.
