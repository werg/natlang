# Qualified direct values in GGUF

The natlang function `Prepare a serving artifact(parent, qualified projection, runtime) -> VerifiedArtifact`
selects the exact qualified parent and records deployment requirements. The existing declared recipe and
artifact manifest carry that policy; this change extends the existing exporter and runtime as exact services.
No second trainer, deployment policy or alternate projection is introduced.

## Current evidence (DGX, 2026-10-14)

`export/gguf.py:export_direct_heads_gguf` still requires `projection.stack`, writes `direct-v1`, and rejects
TokenChoiceProjection. The current NeuraleseProjection has no stack: it is `E[chosen] + C(h)`.
The fork's `tools/neuralese/neuralese.cpp:nz_write_direct` computes a vocabulary softmax mixture plus the
legacy stack residual. Its successful legacy parity certificate cannot qualify the current channel.
The existing parity fixture loads an old artifact without an aligned projection and is skipped when its
artifact environment variables are absent; ordinary green main checks do not cover current GGUF transport.

## Units and exact contracts

1. **Select a qualified artifact**: existing natural-language serving/training policy, with the recipe binding
   exact parent/export/P hashes. Refuse to use an old channel certificate for new weights or projection.
2. **Serialize the projector**: existing crisp GGUF exporter service. Introduce a distinct token-anchored
   profile with declared dimensions, normalisation epsilons, correction linear weight/bias, optional residual
   MLP weights, embedding table and stop IDs. Preserve chosen-token semantics; no mixture fallback.
   Store the optional TCP's existing causal-convolution stack and its exact geometry separately from C.
3. **Load and verify**: existing crisp fork loader. Validate profile, tensor types/shapes, vocabulary and
   finite parameters before pointer casts or allocations. Reject unknown profiles explicitly.
4. **Apply C**: crisp numerical service, because this is the exact trained channel, not a model decision.
   Let x = FP32 RMS(h, eps=1e-6), cast to the linear's computation dtype. Compute linear(x); if present add
   down(GELU(up(RMSNorm(x)))). Anchor is E[argmax(readout(TCP(h)))]; add C and cast as the reference does.
   PyTorch GELU uses the exact erf form by default, not a tanh approximation. The second norm epsilon is
   the actual saved weight dtype's implicit epsilon when RMSNorm.eps is None.
5. **Apply TCP everywhere**: reuse the existing causal-stack implementation, with a separate history per
   sequence and correct reset/copy semantics. It changes both ordinary text and Neuralese token/stop
   choices. Porting only the write path does not qualify an artifact with a trained TCP.
6. **Write and read**: extend existing direct writer and read transport. Writer feedback remains chosen
   tokens; the transmitted values are token anchors plus correction. Preserve scratch sequence state,
   length hints, stop log odds, forced spans and full-depth hidden-state capture.
7. **Verify end to end**: existing parity evaluator, extended to accept the explicit aligned projection.
   Compare nonzero C, nonidentity TCP, history longer than one position, chosen tokens, stop probabilities,
   forced and autonomous payloads, and downstream read logits against the same exact reference weights.
   C=0/TCP=identity is necessary entry evidence but cannot certify the trained path alone.

## Adoption

Start with FP32 CPU parity and exact tensor roundtrips. Then test the actual deployment precision/backend,
including repeated requests and sequence resets. Quantizing the table or changing the backbone requires a
new channel measurement; do not inherit the full-precision certificate. No deployment is qualified until
ordinary text TCP readout and value transport both pass. Any model-facing protocol text change requires
its separate live sample measurement before adoption.
