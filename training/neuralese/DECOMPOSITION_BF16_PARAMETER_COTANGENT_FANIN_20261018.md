# BF16 parameter cotangent fan-in

## Goal and boundary

The same stored BF16 parameter may participate in several operation calls in
one logical update: independent invocations, cached/flat graph branches,
checkpoint replay, and feed-forward or expert chunks. Each operation must keep
the existing stored-weight forward and input-gradient arithmetic. Its weight
cotangent must remain FP32 while those uses are combined, then be converted to
the original parameter dtype exactly once at the optimizer boundary. The
problem is derivative accumulation, not model policy or a new optimizer.

No natural-language policy changes. The existing training recipe still decides
the model, objective, precision mode, and optimizer. This implementation is a
crisp numerical/storage mechanism because exact cotangent arithmetic, alias
identity, bounded spilling, and optimizer handoff are executable invariants.
It must not change checkpoint tensors, optimizer state, RNG, counters, exposure,
forward values, or checkpoint metadata.

## Shared design

1. One operation scope discovers the original trainable low-precision
   parameter objects and aliases before any forward fan-out. It creates one
   scalar FP32 autograd anchor per supported original parameter and one
   scope-owned FP32 cotangent accumulator. All uses of that storage receive the
   same anchor. No model-sized FP32 weight copy or permanent FP32 parameter is
   created.
2. Each supported custom backward computes the weight contribution in FP32,
   routes it to that storage's accumulator, returns no gradient to the
   low-precision weight input, and contributes a scalar sentinel to the shared
   anchor. Existing forward and input-gradient operands remain the exact
   stored-dtype values. The sentinel records that each carrier path ran; it is
   not a surrogate weight gradient.
3. The accumulator adds contributions in operation order in a bounded
   scope-owned buffer. It keeps only the currently active parameter/chunk on
   device and transfers that contribution into an owned CPU/mmap backing store
   in 16 MiB row chunks, avoiding a second full-size host FP32 copy. The full-model 12B case
   must not retain a second FP32 backbone or all FP32 cotangents on GPU. Spill
   files have single-scope ownership and are removed on successful finalization
   or explicit exception cleanup.
4. Two explicit scope modes share the accumulator. A VJP mode lets the anchor's
   autograd backward return the completed FP32 tensor to the original parameter,
   so `autograd.grad` and actor/replay observers see the real derivative without
   mutating `.grad`. An optimizer-update mode retains custom-operation
   contributions across repeated backward calls, suppresses each carrier's
   per-call low-precision parameter gradient, and assigns the once-cast total
   before the unchanged optimizer consumes `.grad`. It adds no trainable
   anchors and preserves optimizer target identity. This mode currently
   supports direct stored-weight carriers; a
   parameterization whose backward transforms a low-precision cotangent must
   provide an explicit shared cotangent rule before it can use repeated
   backward mode.
   Unsupported low-precision uses fail
   before optimizer step instead of silently taking a partial-gradient path.
   Identical `Parameter` objects are one optimizer target; distinct objects with
   overlapping storage are rejected unless their exact view mapping is proven.
5. The existing native weight-carrier interface is the shared entry point for
   stored linears and adapter actor scopes. Ordinary training, shared
   invocation/replay, and actor coefficient differentiation use the same scope
   and accumulator. An invocation view joins the already active operation scope
   for its exact model; it does not open a nested accumulator. A standalone
   actor operation owns one model scope. An unadapted serving operation only
   opens a carrier when an explicitly selected model parameter is trainable;
   frozen inference and no-grad calls keep the existing fast path. Model-target
   VJPs track the requested low-precision base parameters and omit unrequested
   coefficient adjoints, while coefficient-target VJPs retain those auxiliary
   deltas. The operation selection is explicit at the existing gradient API.
6. Adapter composition keeps each actor's exact ordered forward rounding. The
   carrier route sends the shared dW to the base plus the requested adapter
   deltas; same-base actor invocations share only the base accumulator. Model
   parameter targets return the accumulated base derivative through the common
   anchor and do not mutate the previous `.grad` object. For an unadapted target
   request the same carrier is installed before the first module forward.
7. The Maple router's direct FP32 `F.linear` is a separate supported weight
   use: when a low-precision router anchor is active, it uses the shared stored
   linear carrier with the same `x.float()` and `weight.float()` operands as
   its existing route. Dense and grouped expert-bank views retain a full-parent
   accumulator and add through the proven strided view mapping, including when
   the first backward use is a sliced expert. No new route policy or model
   facing behavior is introduced.

## Existing crisp surfaces to extend

- `model/native_weight_carrier.py`: owns the scope, per-storage anchors,
  bounded accumulator, alias validation, finalize/abort lifecycle, and stored
  linear, Conv1d, RMSNorm, and LFM finite-history custom-autograd routes. The
  finite-history carrier preserves the existing explicit FP32 product/reduction
  forward and its stored-dtype input derivative while summing each invocation's
  weight derivative into the common parameter accumulator. Trainable embedding
  lookup uses the same anchor and indexed CPU/mmap row accumulator; each
  occurrence transfers in bounded chunks, while frozen lookups keep native
  `F.embedding` untouched. Mutable `max_norm` and sparse optimizer gradients
  fail explicitly for selected low-precision embedding targets.
- `model/native_adapters.py` and `model/native_actor_graph.py`: pass the active
  scope through functional substitution and checkpoint/recomputation capture;
  retain exact actor identity and coefficient semantics. The no-adapter
  parameter-VJP path still enters the shared scope when its selected BF16
  leaves are trainable; immutable inference remains unchanged.
- `maple/fused_moe.py`, `maple/expert_gemm.py`, and `maple/fp8_experts.py`:
  route invariant, grouped and FP8 expert weight gradients to the same
  accumulator while preserving the existing forward, dX, routing and FP8/QAT
  behavior. No grouped path may cast each expert or chunk contribution to
  BF16 before shared fan-in.
- `maple/model.py`: route the direct low-precision router matrix and dense
  expert-bank views through the same mechanism; preserve FP32 router operands
  and exact top-k behavior.
- `model/shared_state.py` and `model/lfm2_port.py`: cover LFM's two crisp
  convolution execution paths: layer-major finite histories used by packed
  training/replay, and the standalone port's ordinary/fused causal Conv1d.
  LFM's finite-history route does not call the Hugging Face short-convolution
  module forward, so it has its own carrier at the actual reduction site.
- Native training/backward coordinators: bracket one actual optimizer update
  with the scope, including all invocations, checkpoint recomputations,
  row/FFN chunks and repeated loss backward passes. Finalize immediately before
  the existing optimizer consumes gradients. Preserve optimizer and checkpoint
  contracts.

## Required proof

Use real BF16 weights and actual supported operation implementations, not fake
parameter tensors. First compare one use with split invocations, cached versus
flat branches, checkpoint replay, FFN chunks, grouped experts, invariant
experts, and FP8/custom-kernel routes against an FP32 cotangent reference while
requiring stored forward and dX parity. Exercise both LFM and Mellum module
families, aliases/tied storage, repeated backward passes in one update, spill
and cleanup, cancellation/recompute, and strict rejection of any unsupported
low-precision gradient path. Then run the actual CPU complete-gradient
comparison that exposed Mellum's 166/315 BF16 derivative failures; the
per-parameter metrics must pass the existing strict tolerances without
loosening them. CUDA tests must cover each selected fused backend where its
runtime is available. Compare parameter/optimizer/RNG/exposure checkpoint
bytes across a no-op scope and a save/resume continuation. No cloud runtime or
training cutover is part of this change.

## Why the anchor is scalar

A full FP32 copy of every BF16 parameter or a full GPU cotangent mirror violates
the 12B/H100 resource budget. A scalar anchor supplies one common autograd edge
per original parameter before fan-out; custom kernels keep the actual matrix
cotangents in the external bounded accumulator. The original BF16 tensor remains
the value used by forward and dX. This design succeeds only if every low-
precision weight use is intercepted or explicitly rejected, including views,
replay, aliases, grouped experts and all model-specific kernels.
