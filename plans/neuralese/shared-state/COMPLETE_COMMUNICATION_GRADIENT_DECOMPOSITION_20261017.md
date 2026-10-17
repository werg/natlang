# Complete communication derivative qualification

The supervising recipe continues useful training while exact qualification
records what has and has not been proved. Existing channel checks compare
producer/consumer outputs, consumer CE, adapter derivatives and native splice
execution. They are real evidence, but they do not prove every trained backbone
derivative. Cross-row causal-prefix sharing is a separate optional operation.

1. Bind the actual model/arithmetic/backend, complete declared production
   trainable inventory, original graph cases and consumer-loss normalization.
   Distinguish frozen vocabulary weights from learned sparse input/output rows,
   low-rank embedding parameters, backbone parameters and input embeddings.
2. Compare all used declared parameter derivatives and connected input
   embeddings between the existing one-pass graph executor and explicit cached
   execution. Record unused parameters honestly; require nonzero derivatives
   through the producer/input, adapter, backbone and learned input/output roles.
   Preserve absolute and relative/cosine gates; tiny absolute gradients alone
   cannot qualify a missing derivative.
3. Keep one full gradient set resident. During the cached backward, temporary
   leaf hooks compare each arriving gradient and release its old counterpart.
   Use autograd.grad exclusively: never accumulate parameter.grad or arm the
   in-backward optimizer. Preserve existing gradients, optimizer, RNG, backend
   state and train/eval/checkpoint modes. No giant disk copy or second full
   model/gradient allocation is needed.
4. Label old adapter-only checks as channel evidence. Full foundation/shared
   readiness requires a passed complete derivative proof, bound to the exact
   trained identity and inventory. A missing or failed complete check prevents
   readiness; it does not erase actual exposure or stop ordinary useful SGD.
5. A new explicit cotangent profile must pass these same checks on complete
   communication graphs, as well as its separate prefix and actual optimizer
   checks. Do not silently replace the active arithmetic or infer qualification
   from an ordinary SFT capacity measurement.

The numbered policy is natural language. Crisp code is limited to exact
inventory/case/identity verification, tensor derivative comparisons, temporary
hook plumbing and durable qualification receipts. Existing trainers, graph
execution, cached execution, objectives and readiness remain the shared
interfaces for both models.

The CPU audit at
`/home/werg/data/dgxops/native-communication-complete-gradients-20261017.json`
demonstrates why the distinction matters: on the real tiny Mellum
three-invocation graph BF16 outputs/loss/input adjoints match bitwise, while
some per-invocation versus full-row BF16 backbone weight derivatives differ.
This is not a full-size FP8 failure claim or a reason to interrupt active runs.

Stateful backends execute the flat forward and its full VJP first. Only then
restore the exact pre-forward backend state and RNG and construct the cached
forward/VJP lazily. This prevents the cached forward from changing scaling
history retained by a flat backward, and avoids retaining both activation
graphs. The outer shared qualification restores backend extras, counters,
execution flags and RNG on success and failure. A stateful-autograd regression
rejects any intervening backend mutation before its VJP.

Ordinary and joint semantic publication use the same complete qualifier under
their declared live backend. CPU fixtures explicitly declare FP8 off; they
cannot certify the production Mellum FP8 default by replacing its loader.
Historical row credits and full checkpoints remain unchanged when a complete
proof is absent or fails. No CPU fixture is a full-size production capacity or
FP8 qualification. The retained-gradient and comparison-pair byte counts are
not peak memory measurements: cached partial accumulations, autograd return
tuples, metric temporaries and actual destination activations also consume
memory and must be measured before a full-size cloud proof.

Actual trained LFM diagnosis (2026-10-17)
---------------------------------------

The full step9478 parent is
`f07a57e3d98224360b8a509f8f410be3de2fd37830e176c989164d2e83ebcb89`
(2,523,898,927 bytes), restored by independent private GET. Its complete
communication check compares 150 declared trainable parameters plus connected
input embeddings. Under the recorded production arithmetic, 100 of these 151
comparisons fail the existing strict absolute/relative elementwise gates; all
are finite. Forward difference is 1.43e-6 and consumer CE difference 1.14e-5.
Selecting IEEE linear arithmetic alone does not resolve this (101 failures).
These are failed readiness checks, not evidence that actual SGD should stop.

Intermediate tracing localizes the first forward discrepancy in layer zero:

- Adapter output, layer input and elementwise squared norm input are bitwise
  equal between flat and cached execution.
- The FP32 `mean(-1)` reduction first differs by 1.164e-10 when invocation
  shapes differ. Reciprocal square root then differs by 3.815e-6, and the norm
  output by 4.768e-7. This occurs with both recorded and IEEE GEMM arithmetic.
- Calling that same norm one token row at a time gives bitwise matching flat
  and cached norm outputs. This diagnostic isolates reduction scheduling;
  it is not a proposed serial implementation for production.

A subsequent local diagnostic reuses the existing shared Triton RMSNorm on
all 45 HF LFM norms, asserting FP32 inputs/weights to preserve their dtype
semantics. The complete check still fails 96/151 comparisons, with forward
1.07e-6 and CE 1.24e-5 differences. Therefore making the first reduction
independent of row count alone is not a complete repair. No replacement norm,
changed threshold, optimizer update, checkpoint rewrite or production
qualification was adopted from these diagnostics. Complete derivative and
actual optimizer-update checks remain necessary; do not describe the failed
checks as a proven model limitation or an established task-regression cause.

Durable owner-host receipts in `/home/werg/data/dgxops/` are
`lfm9478-complete-gradient-diagnostic-root-v2-20261017.json`,
`lfm9478-norm-reduction-divergence-root-20261017.json`, and
`lfm9478-existing-fusednorm-gradient-diagnostic-root-20261017.json`.
The restored parent remains assigned to this unresolved investigation.

Full inherited update localization (2026-10-17, 15:29 UTC)
-------------------------------------------------------

The actual saved optimizer restored exactly by parameter name: 270 tensors,
575,404,264 bytes and all 150 parameter/group bindings. Serialized slot order
may differ after construction; raw slot indices are not parameter identities.
Private shadow updates preserve the source RNG and perform zero source updates.
With ordinary production arithmetic, 136/150 stored-update comparisons fail
the diagnostic zero-absolute/1e-4-relative gates; 94 parameters have moment
differences. The largest Muon stored-delta relative L2 difference is 0.01418.
Every actual parameter has a nonzero update; no lost-update case was found.
These diagnostic update gates are not additional production qualification rules.

Two bounded full-graph diagnostics then use the existing explicit reference
arithmetic (float64 native projection products and visible attention, original
activation storage). This reduces strict gradient failures from 100 to 35/151
and the largest Muon delta difference to 0.00943. Applying the original source
norm independently per token row within that reference reduces gradient
failures to 29/151, but leaves 127 stored-update diagnostic failures and a
0.00931 maximum Muon delta difference. Neither change is a complete repair or
a production optimization. Both preserve source optimizer/RNG state.

This rules out shape-dependent normalization or native GEMM/attention alone
as a sufficient explanation. Continue localization of the remaining arithmetic
and accumulation differences and their amplification through the actual
optimizer's moment update and BF16 Newton-Schulz recurrence. Distinguish stock
backbone Muon from the separate BF16 MuonSR latent kernels; the latter's buffer
dtype is not evidence about the former's saved buffers. Do not relax readiness gates, deploy
serial norms, attribute task failures to these discrepancies without evidence,
or repeat isolated first-layer checks as if they establish full equivalence.
Useful production SGD continues with its real exposure and optimizer history.

Exact reports are `lfm9478-stored-updates-diagnostic-root-v2-20261017.json`,
`lfm9478-reference-arithmetic-updates-root-20261017.json`,
`lfm9478-rowwise-norm-reference-updates-root-20261017.json`, and their compact
`lfm9478-reference-arithmetic-summary-root-20261017.json` in the same operations
directory. The full inherited runs used at most 7.0 GB host memory and 4.93 GB
CUDA allocation; these are local DGX diagnostic measurements, not cloud proofs.

Stock feed-forward arithmetic localization (2026-10-17, 16:00 UTC)
----------------------------------------------------------------

The reference context did not cover stock LFM feed-forward `nn.Linear` calls.
Adding the existing float64 reference product to those calls, while retaining
the previous reference projections/attention and per-row source norms, passes
all 151 complete communication gradient comparisons at the original tolerances.
The exact inherited optimizer is still unchanged: 105/150 stored-update
diagnostics fail, including 88 parameters with moment differences. The worst
Muon delta relative L2 difference is 0.00504 (absolute difference 2.35e-7 at that parameter).
This localizes an additional arithmetic contribution; it does not establish
which subset of reference changes suffices or justify expensive double products
in production. Tiny gradient differences can cross BF16 orthogonalization rounding
boundaries; their downstream effect still requires investigation.

Receipt `lfm9478-stock-ffn-reference-updates-root-20261017.json` records the
actual original parent, all module names, inherited state, preserved source RNG
and zero source updates. It ran in 171.1 seconds with 6.9 GB peak host memory
and 8.41 GB CUDA allocation. This is a diagnostic execution, not production
qualification or evidence of task-quality improvement. Production continues.

Optimizer dtype follow-up (2026-10-17)
------------------------------------

The installed stock `torch.optim.Muon` initializes its momentum buffer with
`zeros_like(p.grad)` and casts the orthogonalization input to BF16 separately.
The shared `muon_kernels.momentum_nesterov` implementation is for MuonSR latents;
its BF16-buffer contract must not be attributed to every backbone Muon buffer.
The exact inherited-state restore above remains valid, but its existing compact
receipt did not enumerate buffer dtypes. Two isolated CPU-only attempts to add
that inventory hit their declared 2 and 4 GiB limits before producing a result,
including the latter's metadata-only storage mapping. They performed no model
execution or source updates and provide no dtype or numerical qualification.
Use the actual optimizer class and saved buffer dtype when localizing the
remaining update amplification; a BF16 backbone-momentum explanation remains
unproved. Keep the assigned original checkpoint and existing full comparison
receipts for that investigation.

Feed-forward-only isolation (2026-10-17, 16:25 UTC)
-------------------------------------------------

Replacing only the 48 stock feed-forward linear products with the float64
reference, while keeping all other production arithmetic, leaves 53/151
complete communication gradient comparisons failing. Forward maximum absolute
error is 1.13249e-6 and cross-entropy absolute difference is 2.86102e-6.
Thus feed-forward products contribute to the discrepancy, but their isolated
replacement does not establish complete equivalence. The combined reference
result above remains distinct from this narrower test.

Receipt `lfm9478-only-stock-ffn-reference-root-20261017.json` records the
unchanged original parent and this exact intervention. The ledgered execution
took 131.86 seconds and used 6.1 GB peak host memory. No production arithmetic,
readiness threshold, optimizer history or source checkpoint was changed.

Fused-norm combined reference and actual optimizer dtypes (2026-10-17, 20:58 UTC)
-------------------------------------------------------------------------------

Replacing the serial source-norm diagnostic with the existing shared Triton
fused RMSNorm, while retaining the explicit double reference projections,
visible attention and stock feed-forward products, passes all 151 complete
gradient comparisons. Forward, incremental and consumer-loss differences are
zero. This demonstrates that serial token execution is unnecessary for that
combined diagnostic result; it does not qualify production arithmetic or make
the expensive double products a production proposal.

The same run restores and verifies the actual step9478 optimizer by parameter
name before observing its tensor dtypes. All 92 matrix momentum buffers belong
to **MuonSR** and are BF16; the 58 AdamW parameters have FP32 first/second moments
and step buffers. This resolves the previous unmeasured route/dtype question
for this exact checkpoint. It must not be generalized to a different saved
optimizer or used to silently convert moments during a resume.

Under the unchanged strict diagnostic stored-update gates, 101/150 comparisons
still fail. Source RNG is preserved, all inherited tensors are checked exactly,
and no source optimizer update occurs. Thus passing complete derivatives under
reference arithmetic does not establish identical shadow updates. Further
localization should use these actual MuonSR BF16 momentum and recurrence
semantics rather than assuming stock torch Muon.

Receipt `lfm9478-fusednorm-combined-reference-updates-root-20261017.json` binds
the original parent, all module names and per-parameter buffer inventories.
The ledgered run took152.1 seconds, allocated at most8.40GB CUDA and completed
without an exception. Existing active student training and readiness thresholds
were unchanged.

Cheaper IEEE/fused-norm combination (2026-10-17, 21:24 UTC)
---------------------------------------------------------

Using IEEE native projections/attention and the existing fused RMSNorm with
global float32 matmul set to highest and TF32 disabled still fails98/151 full
gradient comparisons. Forward maximum difference is1.788e-6, incremental
difference8.94e-7 and consumer CE difference1.526e-5. The recorded global
settings were already highest/TF32-off before the intervention. Thus a missing
global precision flag or this inexpensive combination does not explain the
remaining discrepancy; no production arithmetic or tolerance was changed.

Receipt `lfm9478-all-ieee-fusednorm-gradient-diagnostic-root-v2-20261017.json`
binds the unchanged original parent and source RNG. The ledgered run completed
in104.8seconds with6.5GB peak host memory and3.85GB CUDA. The preceding attempt
failed its initial CUDA model allocation before any comparison and remains an
infrastructure failure, not numerical evidence; the explicit retry required
12GiB free driver memory before loading. Further work should localize the
remaining kernel/adjoint discrepancy rather than repeat this combination.
