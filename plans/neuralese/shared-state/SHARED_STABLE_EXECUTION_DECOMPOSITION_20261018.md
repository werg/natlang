# Stable execution of shared-state graphs

The production policy is to repair the localized numerical defects and keep training complete graphs, without relaxing their exact cached-reference checks. The actual 9478 checkpoint passes all 151 derivatives in graph/cached comparisons and the 4,099-token FFN 0→4,096 transition when existing fixed IEEE FFN kernels, existing fused row RMS kernels and wider convolution-state fan-in are combined. The failed 12756 cloud transition remains failed until that actual parent is checked.

1. Add one explicit shared execution profile, with absent/`source-v1` retaining legacy behavior and `stable-v1` selecting the repaired mechanisms. Record it in checkpoint configuration, native identity, trainer arguments/history and shared recipe inheritance. Restoring or switching it must preserve exact weights, optimizer moments, RNG and completed exposure.
2. Reuse existing invariant linear and fused RMS kernels. Stock LFM FFNs need fixed row forward/input adjoints; Mellum's custom expert implementation remains its existing declared implementation. Apply the supported RMS rule through the existing module objects, retaining functional adapter substitution and CPU reference behavior. Do not copy kernel implementations or mutate global torch precision.
3. Carry differentiable convolution source edges in FP64 until shared fan-in is accumulated. Use a bounded gather with FP32 history values and FP64 source accumulation, avoiding a dense FP64 history expansion. Ordinary inference remains FP32 state values; its repaired FFN/norm forward rule is the same. Preserve this across the one-pass graph, cached reference, native cached prefill/decode, differentiable export/import and actor scopes. Architectures without convolution acquire no invented state.
4. Before a changed profile trains, run the existing exact cached reference with the full actual production trainable inventory, under the actual backend and unchanged tolerances. Record that explicit transition separately from legacy certificates. Existing memory qualification must bind the new profile; it must execute complete backward before SGD. An old failed gradient proof is not reinterpreted as a pass.
5. Verify both architectures, checkpoint recomputation, real optimizer/RNG/exposure continuation, block transport/cached inference and fixed/unchunked FFNs. Compare bounded gather against its mathematical reference, including duplicate/shared rows, masked gradients and multiple consumers; measure actual GPU peak to reject a hidden FP64 history expansion.
6. Independently review the integrated code before cutover. Qualify the actual current parent and full policy capacity; resume useful established-capacity SFT while this implementation proceeds. The small diagnostic and publishing source alone grant no production readiness.

Checkpoint-control integration correction: raw-policy preparation deliberately
uses saved embedding/interface tensor views without constructing a backbone.
Separate declared-profile/tensor identity from actual module-dispatch validation.
Both paths share the exact embedding identity law; served models still validate
their installed modules. Checkpoint-only preparation validates the saved shared
profile against its configuration and does not claim runtime qualification.
Carry the shared execution profile in new interface snapshots as well as model
configuration, rejecting disagreement. Existing source-profile snapshots remain
byte-compatible. A missing field in an earlier stable diagnostic interface is
interpreted using its already explicit model configuration, not a new profile.
Validation: 171 focused tests passed, covering all 45 previously failing raw
policy/control cases, both architectures and all three profile declarations,
no second model construction or parent decode, exact RNG preservation,
source-to-stable continuation, explicit interface/config disagreement and actual
installed-kernel tampering. Independent source review cleared the split.

This is crisp arithmetic, durability and verification plumbing for the owner's natural-language supervision policy. It adds no new model-facing markers, task policy, scheduling or latent channel. The full training and inference path, source/continuation identities and cross-model compatibility checks must land together; do not publish a disconnected helper as completion.

Integrated validation (2026-10-18): the historical 9478 checkpoint passed the complete 4,099-token FFN transition with the bounded gather implementation (`lfm9478-integrated-stable-ffn4096-transition-root-20261018.json`, 291 seconds, no SGD, unchanged tolerances). This comparison uses fresh diagnostic optimizer inventory and grants no current-parent readiness. Independent review then caught and repaired private RMS gain preservation and the higher-order reference fallback. The expanded suite passed 140 cases, including both architectures on actual CUDA, every trainable external-state gradient, checkpoint recomputation, private gains, second derivatives, and exact saved weights/moments/RNG/exposure across a source-to-stable continuation. The source-profile CUDA fixture explicitly uses production's declared stable norm dispatch; raw Maple constructor compilation donates backward buffers and forbids its retained-graph comparison. Independent integrated review cleared sealing; actual latest-parent qualification and complete policy capacity remain mandatory before deployment.
