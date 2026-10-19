# Native inference throughput

Policy remains the existing natlang program's choice of calls and communication.
`advanceNativeInvocation(state, sampledAction) -> invocationState` is exact crisp
execution: token sampling, positional KV/hybrid updates, authenticated native
captures, and full episode provenance retain the existing contracts.

1. Profile the actual shared prefill/decode mechanisms before choosing work.
2. Reuse immutable one-query block geometry while rebinding exact causal/window
   bounds and real sequence lengths. This is exact kernel plumbing, not another
   attention algorithm; preserve chronological partial-block accumulation.
3. Ordinary-token inference can omit a known-zero low-rank residual only when
   autograd is disabled. Training/reference gradient and optimizer contracts stay
   unchanged; producer inputs still use the trained transform.
4. Execute a known forced ordinary template prefix as one native prefill chunk,
   retaining every forced input/action coordinate and actual hybrid-state result.
5. Validate both model ports against the previous cached execution, including
   imported producer state, window boundaries, captures and gradients. Measure
   the trained actor only in a bounded owner-coordinated local window.

No new prompt, reward, scheduling policy, distribution or communication format
is introduced. No projection fallback, fabricated cache, recursive graph replay,
cloud allocation or unmeasured production throughput claim is permitted.

Current validated implementation: steps2–3 only. Template-prefix batching in
step4 remains an assigned followup: a focused CUDA check found a small difference
already after the first LFM convolution/FFN layer; CPU both-port checks were
exact. The existing sequential prefix is retained pending localization and
appropriate actual-actor qualification; no tolerance was widened.

Unadapted no-grad execution also omits the per-round parameter inventory walk
used only to decide whether an autograd cotangent carrier is needed. Functional
adapters still resolve/substitute exactly; gradient-enabled VJPs retain the same
parameter discovery and carrier routing.

Measured next mechanism: fixed linear single-query inference currently pads to
64 rows and accounts for about86% of actual trained LFM GPU time. Reuse the same
grouped dense kernel with a16-row inference tile for at most16 actual rows; keep
K/N tiles, dot precision and accumulation order. Ordinary training/adjoints keep
the existing64-row tile. Before adoption compare exact outputs and explicit
input adjoints across arithmetic modes, then full fan-out graph gradients on both
canonical ports and actual trained-actor sampled/captured-state parity. No
precision downgrade or probability-law change is authorized by this optimization.

Next measured launch work preserves one shared invariant-linear forward. True
inference calls it directly without AutogradFunction save/backward bookkeeping;
all gradient-enabled and ordinary no-grad calls retain the original Function.
A single-expert inference kernel writes every real output element, so its output
can be allocated without a redundant zero fill. Grouped expert padding and all
training/backward allocations retain their existing zero initialization. Poisoned
allocation, storage-cast/adapter parity, both-port captures and exact gradient
checks qualify these crisp changes before another full-size timing window.
