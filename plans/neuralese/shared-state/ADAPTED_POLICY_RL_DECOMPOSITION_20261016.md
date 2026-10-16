# Adapted native actor policy replay

The service-facing natural-language operation is: retain the exact actor used for each invocation, reconstruct its actual producer/consumer graph, and improve that same policy from verified task feedback. A caller binds immutable native coefficient artifacts and scales with `withAdapters`; coefficients are not communication state. Retain the base checkpoint and complete adapter basis/coefficient dependency closure before returning durable sampled-policy references.

Exact helpers validate ordered adapter bindings, effective actor identity, archive bytes and every sampled input/query coordinate. They compile the existing one-pass shared-state graph, then run its existing policy objective and backward under the same functional substitution used by generation and coefficient gradients. Activation-checkpoint recomputation remains inside that scope. Consumer loss must reach replayed producer coefficients; archived producer KV is evidence, never a substitute for that gradient path.

1. Capture native sampling with an explicit adapter-bound schema, base identity and canonical ordered coefficient/scales binding. Reject changes within an invocation.
2. Archive named artifacts recursively alongside exact native state using the existing RL bundle and immutable tensor codec. Verify all content IDs, bytes and dependency identities on restore.
3. Compile only a homogeneous exact actor cohort through the current shared executor. Reject heterogeneous bindings explicitly; do not apply one adapter globally to differently adapted calls.
4. In the native gradient service, validate the admitted objective against the active effective identity and exact adapter binding. Evaluate the existing packed graph loss and its coefficient/parameter gradients inside the existing functional operation.
5. Test actual both-port capture, cached versus graph log probabilities, consumer-to-producer coefficient adjoints, durable replay after store restart, identity/scale/archive tampering and unadapted isolation.

Heterogeneous per-invocation adapters remain an explicit executor extension: physical own rows must carry invocation bindings, projections/FFNs must apply their exact binding at each layer while imported aliases retain producer KV, and the single forward/backward must gather results in original physical order. Recursive invocation replay and blanket global substitution do not implement that contract. Full-backbone production trainer admission of an adapted actor also requires preserving the adapter scope through backward and exporting/rebinding its changed base; it must not silently label the effective actor SHA as a checkpoint-file SHA. Until those paths are implemented and qualified, capabilities stay unadvertised and production actor/controller configuration is unchanged.

## Implemented fixed-actor boundary and evidence

Implemented: adapted sampling/3 receipts; immutable ordered actor binding and
mid-invocation mutation rejection; real producer occurrence aliasing; exact
coefficient-policy gradient objective under the shared functional operation;
recursive named/native RL bundle archives; durable store reopen; native HTTP
gradient/Adam update; continued actor bindings preserve every ordered occurrence,
scale and untargeted adapter. Generic named artifact retention/archive shares one
dependency helper in each language. Direct Engine.generate now prepares the same
prompt plan as submit (previously native direct generation always failed with an
empty prompt).

Terminal ledger evidence (logs under `/home/werg/data/dgxops/`):

- `native-adapted-rl-final-verified-20261016`:19 PASS22.21s, actual both-port
  Engine capture, strict FP32 graph/cached logps and coefficient adjoints,
  independent cached oracle, V-adapter detached-writer counterexample, actual
  native HTTP gradient/Adam, repeated/scaled multi-adapter continuation, durable
  reopen and hash tamper rejection, existing native serve/replay/bundle paths.
- `native-adapted-rl-node-final-20261016`: TS build,2 actual typed HTTP adapter
  archive/learning/portable restart tests PASS10.34s, quick PASS.
- `native-adapted-rl-cuda-final-20261016`:2 strict FP32 both-port CUDA tests
  PASS22.31s before final deterministic fixture binding; final bound rerun
  `native-adapted-rl-cuda-bound-20261016` PASS2cases9.90s.
- The initial actor tests exposed missing direct prompt preparation. Later
  strengthened tests correctly rejected a final-layer Q fixture as evidence of
  writer contribution; V explicitly affects exported state and now proves it.
  No failing test was skipped or tolerance widened.

### Explicit remaining qualification and implementation gaps

Mellum production is BF16. The FP32 diagnostic explicitly promotes an isolated
copy and derives a new diagnostic identity; it does not use or weaken the native
loader's BF16 source guard. Actual BF16 serving, archive, policy-gradient and
Adam durability pass, but BF16 packed/cached coefficient-adjoint parity does
NOT inherit the FP32 certificate. The retained failed oracle run
`native-adapted-rl-scope-20261016` observed1.87e-4 max absolute /~1.1% maximum
coordinate relative discrepancy. Deterministic BF16 localization in the final
verified run measures0.003834746 relative-L2 coefficient gap and0.003094903
relative-L2 effective-weight cotangent gap. An exact FP64 adapter Jacobian
accounts for the coefficient discrepancy from the actual BF16 weight cotangent
with4.22118e-5 relative-gap residual. This localizes the discrepancy upstream of
the adapter coefficient algebra; it does not establish which earliest backbone
operator introduces it. Shape-dependent BF16 weight-gradient accumulation is a
hypothesis requiring exact operator localization, not a waived parity gate.

Heterogeneous per-invocation bindings, full-backbone adapted actor export/rebind,
production controller actor attestation for configured adapters and BF16 adjoint
qualification remain OPEN. No capability or production recipe is enabled by this
commit. Existing production arithmetic and raw/semantic actor deployments remain
unchanged. Follow-up must extend layer-major own-row projection/FFN binding while
retaining imported producer state and one forward/backward, and must preserve
checkpoint/optimizer/RNG state when exporting a changed adapted backbone.

- `native-adapted-rl-mutation-20261016`: actual scheduler prefill then mutated
  adapter scale rejects before the next token;1PASS.

Next numerical mechanism investigation: an explicit FP32 effective-weight
cotangent carrier with unchanged BF16 forward storage. Existing invariant
linear backward already computes FP32 dW, but a BF16 autograd input casts it
before the coefficient chain. The fix must preserve forward weight bits and
executed kernels, extend shared linear/expert plumbing, and qualify actual
cached versus graph coefficient adjoints before any production opt-in.
