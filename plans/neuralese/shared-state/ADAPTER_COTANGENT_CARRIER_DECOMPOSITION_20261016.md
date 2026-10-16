# Native adapter coefficient cotangent precision

Natural-language contract: learn the selected immutable native adapter from the
same actor's actual behavior while preserving that actor's exact forward
weights and kernel arithmetic. A lower precision storage format must not erase
coefficient learning merely because the effective dense weight is materialized.
The optional gradient setting is operational arithmetic, not a new policy or a
claim of task improvement. Existing defaults remain unchanged pending evidence.

Exact plumbing needs separate forward storage dtype and autograd carrier dtype:

1. Form effective weights with the existing ordered per-adapter casts and sums.
   For the opt-in FP32 carrier, retain exactly those BF16-rounded values in an
   FP32 autograd tensor. Its straight-through derivative accumulates coefficient
   cotangents before a BF16 storage cast can erase or repartition them.
2. Use the existing native fixed linear kernel for attention/conv projections,
   and original Torch linear forward for ordinary HF FFNs. Selected Linear
   modules receive an exact temporary wrapper with the same parameter names,
   ties and original objects restored on every exit. This is functional native
   substitution, not projected communication or legacy AdapterBank hooks.
3. Each primitive casts only inside its custom-autograd forward to the declared
   original storage dtype, executes the unchanged forward/dX path and retains
   FP32 dW for its carrier input. Nonadapted and default-storage calls retain
   their original code paths. CPU implements an exact Torch oracle.
4. Expert banks retain carrier values through DenseExperts.stacked/unbind;
   invariant/grouped/FP8 projections use the same forward storage dtype and a
   shared GPU FP32 grouped dW accumulation helper, avoiding CPU expert loops.
   Loop execution uses the same storage-aware Torch primitive on bank views.
5. Requests/artifacts explicitly record the selected coefficient-gradient
   precision. Exact actor identity remains its existing forward binding because
   forward bits do not change. Optimizer moments and coefficient artifact
   identities retain their existing durable lifecycle.

Acceptance requires bitwise unchanged forwards (base→adapted→base included),
strict cached/one-graph coefficient adjoints, exact FP64 affine oracle, actual
both-port attention/FFN and Mellum expert paths, actual executed FP8 projections,
complete checkpoint-recompute scope, restoration on failure and direct service
learning. Do not advertise partial target inventories. Heterogeneous invocation
bindings and changed-backbone actor export remain separate open executor work.

## Implementation and qualification scope

The explicit native gradient request selects `adapter_gradient_precision` with
`storage` default or `fp32`. The typed client exposes `adapterGradientPrecision`.
Artifacts record this choice; neither actor forward identity nor optimizer
history is rewritten. `rounded_weight_carrier` preserves each ordered adapter
cast/add before retaining its values in FP32. Transient selected Linear wrappers
preserve hooks, aliases and original parameters. Native fixed projections and
ordinary HF FFNs keep their different existing forward kernel contracts.

All expert dispatches (invariant, grouped with/without glue, loop and executed
FP8) preserve exact original forward storage bits. `grouped_weight_gradient`
uses GPU group offsets without host synchronization, writes FP32 and handles
empty groups and odd dimensions. Dense linear carrier dW uses the existing
fixed native kernel with FP32 output, avoiding cuBLAS environment TF32 overrides.
This is separate from the existing optional FP8 input-adjoint recipe; both
`current` and `current-bf16-dx` are exercised without changing either recipe.

The combined CUDA dragnet exposed another real shared attention cache bug:
`_compiled_flex` did not distinguish Q/K/V adjoint masks and inference/autograd
modes. Multiple legitimate target selections exhausted one wrapper's8Dynamo
specializations. Its exact cache contract now includes those flags and actual
inference tensor state. Sequence lengths remain dynamic. No compiler limit was
raised, no cache reset substituted for correctness, and no dense fallback was
introduced. A real combined16-case CUDA run passes after this repair.

Terminal evidence under `/home/werg/data/dgxops/`:

- `native-adapter-carriers-cpu-20261016`:6PASS15.37s, actual native both ports.
- `native-adapter-carriers-cache-cuda-20261016`:16PASS58.10s. Six actual native
  both-port Q/V/FFN/expert cases; six actual executed FP8 current and BF16-dX
  cases, with increased executed projection counts in each phase; FP64 grouped
  weight-gradient oracles for BF16/FP32 with empty/odd groups; actual HTTP
  gradient; failure restoration and compiler-cache contract. Checkpointed graph
  backward runs inside the full backend/adapter scope. Forward hidden/logits/
  logps remain bitwise identical to the storage baseline. Cached coefficient
  adjoints pass unchangedatol3e-6/rtol2e-5 and globalrelativeL2<2e-5.
- `native-adapter-carriers-grouped-20261016`:2PASS10.33s, actual BF16 grouped
  execution with glue disabled/enabled, custom projection calls observed,
  bitwise baseline forward and real finite coefficient gradients.
- `native-adapter-carriers-regression-20261016`:39PASS88.92s, existing serving
  adapters, policy capture/replay, native precision and actual CUDA continuation.
- `native-adapter-carriers-build-20261016`: TS build and quick PASS.

Local evidence uses Torch2.11/cu130/SM121 and tiny real architecture fixtures.
It is not an H100/NGC2.14 restored-production-parent certificate or a task-quality
claim. Production defaults/capabilities/recipes remain unchanged. Independent
root review and any concrete target-parent qualification precede adoption.
Heterogeneous invocation weights, changing-backbone adapted actor export and
multiupdate behavior/current coefficient binding remain separate open work.

Additional focused coverage:

- `native-adapter-carriers-ffn-bf16-20261016`:2PASS15.90s, actual BF16 LFM
  stock HF FFN up/down projections, bitwise forward and strict cached adjoints.
- `native-adapter-carriers-ordered-20261016`:3PASS18.92s, actual CUDA repeated
  scaled coefficients preserve every ordered forward rounding and sum FP32
  adjoints; both actual policy HTTP+Adam+durable bundle cases now explicitly use
  FP32 coefficient cotangents, including scaled multi-adapter state.

The original combined run before the cache repair was14PASS/1FAIL on Dynamo's
specialization limit (`native-adapter-carriers-final-cuda-20261016`); it is not
reported as green. The final combined16PASS is the repaired mechanism evidence.
