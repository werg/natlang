# Retained behavior and evolving adapter policy

Natural-language service contract: improve the current adapter policy from the
same admitted rollout across useful optimizer steps. Preserve the behavior
policy, sampled-token log probabilities, rewards and archived observations as
facts. Distinguish them from the coefficients currently being optimized. A
changed coefficient ID is expected after an immutable update; it must not erase
or relabel the original behavior receipt.

Exact verifier/plumbing:

1. The objective retains its original actor identity and ordered coefficients,
   scales, graph/source bindings and immutable archived coefficient/basis bytes.
2. Bind current evaluation coefficients explicitly inside the native functional
   scope. Require the exact same base/tokenizer/embedding/architecture and each
   ordered scale, matrix specification and frozen basis. Coefficients may evolve;
   unrelated bases, architectures, bases-of-adaptation or reordered scopes fail.
3. Reconstruct internal producers with current coefficients in the existing
   single graph forward. Keep genuine external observations at their actual
   behavior identity. Never rename old wire artifacts to current IDs or detach
   internal producer paths into archived KV.
4. Evaluate existing GRPO clipping against retained behavior logps, differentiate
   the actual current policy, and publish an immutable coefficient update with
   existing exact-parent optimizer state. Gradient provenance names both actors.
5. Test multiple real service gradient/Adam steps on one admitted objective,
   exact moments, objective/logp preservation, durable restart and adversarial
   scope/basis/scale mismatches. Keep changed-backbone actor export and mixed
   per-invocation policies explicit separate work.

## Strict transport boundary before explicit same-base mode

Internal replayed producers support current-coefficient continuation. First-step
external observations still bind the exact actor. Once coefficients change,
default strict Engine serving would text-reencode old actor KV; replaying that
KV unchanged is only a fixed-observation surrogate and is NOT actual-runtime
parity. This commit therefore rejects such multiupdate external inputs before
executing a model forward. The prior four-case diagnostic proved exact Adam
mechanics for fixed observations but is not a runtime certificate. The next
explicit caller-declared same-EXACT-base adapter-state splice mechanism must
cover serving, per-input graph identities and all receipts before reopening
that path. Different-base/model compatibility is not inferred.

Focused ledger evidence (`/home/werg/data/dgxops/<unit>.log`):
`native-policy-behavior-continuation-20261016`21PASS28.08s includes the initial
fixed-observation diagnostic, legacy graph/replay tests; its external changed-
actor cases are NOT actual-runtime qualification. Likewise the initial CUDA
four-case diagnostic PASS35.92s proves numerical/optimizer mechanics only.
After adding the strict transport guard,
`native-policy-behavior-strict-20261016`4PASS25.13s verifies two-step exact Adam
for internal graphs on both ports and explicit rejection for changed-policy
external KV. Default strict transport now cannot silently train the fixed-KV
surrogate as though it were current serving behavior. Original objective bytes,
behavior values and optimizer state are preserved; both actor identities are
recorded in gradient provenance. Explicit same-base transport is the immediate
next implementation, not declared complete here.
Final guard/recovery text and mechanics:
`native-policy-behavior-publish-20261016`4PASS25.03s and quick PASS.
The standing root service owns the broad integrated suite; no duplicate full
suite was launched for this focused repair.
