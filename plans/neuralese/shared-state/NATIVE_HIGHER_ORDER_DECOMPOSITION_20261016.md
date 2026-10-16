# Native higher-order learning

The service-facing natlang operation is: learn how a declared sequence of
learning updates changes a later task objective. The caller names the exact
native parameters, adapter coefficients or consumed shared-state tensors,
then supplies a finite dependency DAG of gradient observations and optimizer
updates. The service returns identity-bound named adjoints. It must never
interpret projected legacy values as native state or infer an actor from shape.

Policy is the existing learning program: select the task, learning targets and
update rule, evaluate the later objective, and improve the chosen inputs. No
new hidden policy or teacher-generated labels are introduced here. The exact
DAG verifier, tensor substitution, mathematical differentiation, immutable
artifact transport and atomic publication are crisp service mechanisms.

1. Authenticate the current base, actors, graph rows and imported-state IDs.
   Verify every named target and each finite derived dependency before forward.
2. Execute each native graph once using its existing layer-major plan, role
   masks, causal imports, convolution history and objective normalization.
   A subsequent optimizer update is a genuine separate mathematical step,
   never a recursive replay of producers for a consumer import.
3. Use an explicitly requested and reported reference arithmetic contract for
   higher-order learning. Current CUDA Flex/AOT attention and several custom
   linear/expert backwards do not implement double backward. Their Hessian is
   not claimed. Ordinary first-order serving/training kernels remain default.
   The reference path uses differentiable torch primitives, the same exact
   state visibility, bounded query tiles, stored activation casts and explicit
   high-precision matrix products. It does not attach a synthetic gradient to
   a detached fast-kernel result. Forward values and first adjoints belong to
   this declared arithmetic mode and are measured against its actual reference.
4. Replay a finite derived gradient/update DAG under that same contract.
   Bind every node to its actual input/output artifacts and optimizer history;
   reject missing, stale, reordered or incompatible dependencies. Preserve
   named moments, step counters and all unrelated model state. Do not reuse a
   first-order observation as an unrecorded differentiable computation.
5. Differentiate the outer objective through all declared updates. Return named
   artifacts carrying the arithmetic contract, exact actor/graph/dependency
   bindings and target inventory. Archive and restore the whole dependency
   closure through the existing tensor store and typed client lifecycle.
6. Restore module objects, parameter/buffer storage, grad flags, RNG, backend
   state and scopes after success or failure. Advertise higher-order support
   only after the actual service/client/archive and both native ports pass.

Units: the typed request/result and finite dependency DAG become the learning
service contract; reference arithmetic is one shared crisp backend; functional
optimizer equations extend the existing named optimizer mechanism; source/task
selection remains the invoking natlang program. Existing first-order native
gradient execution, artifact codec and actor substitution are reused rather
than a parallel projected GradSession. Per-model exceptions are limited to
existing attention/RoPE and LFM hybrid-state mechanisms.

Qualification remains open. Required evidence includes both actual native
ports on CPU/CUDA, authentic producer/consumer masks and hybrid state, all
named target classes, analytic/finite-difference HVP references, repeated
SGD/Adam-derived updates, immutable transport/reload, failure restoration and
exact preservation of ordinary first-order behavior. Fullsize cloud execution
is a separate operational qualification; tiny fixtures cannot establish it.

The initial arithmetic mechanism is internal and unadvertised. Its binding
specifies FP64 native projection/attention/expert products, source activation
storage, source-dtype differentiable stock torch modules, and stored base
weights plus unrounded FP32 adapter deltas. It never claims the rounded fast
actor's first adjoint or Hessian. Query tiles bound score workspace without
changing causal visibility or truncating source rows. No full model copy or
persistent FP64 weight bank is created. Checkpoint callbacks explicitly capture
the reference scope because CUDA recomputation threads do not inherit Python
ContextVars; reuse after the operation closes is rejected.

Functional service updates retain the existing named moment/counter inventory
and use upstream torch optimizer primitives. Native dampening-zero SGD starts
with an explicit zero momentum buffer: identical first values, without torch's
otherwise detached initial gradient. Adam uses the ordinary scalar bias
correction path under enabled autograd rather than its numerically different
FP32-counter differentiable branch. Its nonnegative moment root has an explicit
zero-v boundary tangent of zero; along a valid zero history this gives the
finite composite derivative through epsilon, instead of an artificial NaN from
zero times sqrt's infinite derivative. All positive-v values and tangents use
the ordinary root; no detached observation or gradient replacement is attached.

A multi-update program retains one operation lifetime across inner gradient
scopes. Checkpoint capture under this explicit reference operation retains exact
current parameter/buffer tensor references and re-enters their functional module
bindings when a later outer backward recomputes an earlier layer or readout.
This preserves the earlier effective actor after newer updates become active,
without cloning the whole model or recursively replaying communication calls.
Ordinary first-order scopes still close at their original boundary. The reference
lifetime expires at final publication/failure; later recomputation must reject.
Both uniform and heterogeneous producer/consumer actors require this mechanism.

Internal qualification: actual CPU mathematical attention/linear gradgradcheck,
real native all-parameter double backward, coefficient and external KV/hybrid
leaf derivatives, three-step optimizer values/moments plus numerical analytic
hypergradients, and two native SGD/Adam updates before the outer derivative.
Checkpointed and uncheckpointed hypergradients must satisfy global relative-L2
and cosine checks as well as elementwise checks; tiny coordinates alone cannot
pass on absolute tolerance. These are engineering checks of the internal backend,
not completion of the still-open HTTP/typed derivation-program/archive contract.

### File-backed checkpoint interaction qualification

The same two-update native reference-arithmetic oracle now runs with exact
file-backed checkpoint inputs: both ports, SGD and Adam, and uniform and
heterogeneous writer/reader actors. It compares final loss, actual updated
coefficients and nonzero outer root gradients against uncheckpointed execution,
including relative-L2/cosine checks and actual file pack/unpack counters. The
12:45 local CPU and CUDA runs passed all 16 cases each (CUDA 7.90 s), using the
stride-preserving storage implementation 72957ded9. This qualifies the native
higher-order interaction; it does not qualify production whole-graph capacity
or expose the unfinished public derivative-program API.

### Finite program and durable optimized-state contract (implementation underway)

The service-facing unit is a finite typed derivation program: named native
parameter, adapter or imported-state inputs; ordered gradient and immutable
optimizer nodes; and an outer objective differentiated with respect to named
original inputs. Natural-language callers choose the objective, updates and
stopping structure. Crisp code verifies acyclic references, exact artifact and
model identities, executes each objective through the existing masked native
graph executor, and applies the shared named service optimizer. A communication
graph still has one masked forward; optimizer nodes intentionally create new
forward evaluations at changed values. This is not recursive communication replay.

1. Bind an immutable program declaration before execution, preserving its exact
   JSON bytes and arithmetic contract. Root input and static artifact dependencies
   are explicit. Output IDs occur only in the later execution receipt, avoiding
   cycles in content hashes.
2. Materialize every actual changed adapter or native state under its real ID
   before using it. Retain live tensors separately for automatic differentiation;
   durable artifacts contain the exact evaluated values, never placeholder actors.
3. A native state update preserves the original producer as origin and records
   the parent state, actual gradient, numeric update, optimizer history/step,
   algorithm settings and optional derivative-program ID. Original text, token
   IDs, embeddings, positions and layout remain unchanged. The consumer verifies
   archived parents and exact selected numeric payloads; it must not describe the
   result as a fresh producer capture.
4. Preserve per-name optimizer moments and counters across successive nodes.
   Missing or incompatible histories fail; an explicit new history starts at
   step one. No optimizer, model, global RNG or original artifact is mutated.
5. Python and TypeScript stores, portable archives, inference receipts and the
   public learning API must preserve this same derivation graph. A graph-only
   endpoint is an internal component, not completion of public GradSession support.

This next protocol is not advertised until all paths and both-port service
qualification are complete. The mathematical reference component and offload
interaction above are qualified independently of that unfinished integration.

Optimized values also cross semantic adaptation: a semantic encoding records its
exact optimized raw parent and adapter SHA; a decoded target state records the
source artifact, source/target model identities and trained adapter SHA. These
are explicitly semantic transformations, never producer captures. The decoder
runs the existing SemanticChannel mechanism; no alternative projection or
text-only replacement is introduced. Portable validation checks immutable
provenance and unchanged source fields, not a claim of offline numerical VAE
re-execution. Actual constructor/consumer equivalence remains a separate test.

Internal finite-program component qualification (2026-10-16): both ports pass
two actual SGD/Adam updates with root meta-gradients, explicit same-arithmetic
manual autograd oracles, named backbone and KV/hybrid roots, heterogeneous
actors, unused-input zero derivatives, and exact original-versus-evaluated
actor separation. A gradient records a complete functional actor descriptor:
base identity, actual parameter-set overlays, uniform or per-invocation
coefficients and reference arithmetic. It never reuses a coefficient-only
checkpoint hash for changed backbone values. Immutable declaration/execution
bindings are verified before accepting program-bound gradients.

The focused final CPU run passed 35 cases (14.11 s); actual CUDA passed all
28 program/state cases (27.78 s, 2.7 GB peak). These include actual typed HTTP
two-step imported-state optimization, restart/restore, Python-to-TypeScript
File/Memory/OPFS archives and `.nz` round trips, forged actor/dependency
rejection, and optimized-state semantic encode/decode provenance. The existing
14 native-store/tensor/OPFS tests also passed. This does not claim full-size
cloud capacity, task lift, or public automatic higher-order support.

Remaining public integration is explicit: recorded native turns must preserve
actual native sampling/producer receipts; the existing `GradSession` objectives
must lower to native masked graphs and the finite DAG; registered-policy
objective files need portable dependency closure. Public order-2 HTTP support
and capability metadata remain guarded. Persisted parameter-set roots currently
must match the actual live checkpoint; within-program changed parameter sets
are supported and individually identified. Exporting a new serving checkpoint
from an arbitrary changed parameter-set root is a separate operation.

### Independent-review repair: numeric update versus immutable payload proof

An immutable update artifact proves which bytes a state contains; it does not
alone prove those bytes followed the declared optimizer. Python now reexecutes
the existing `functional_named_service_step` on the selected small state tensors
and rejects a jointly forged update artifact plus matching state. It also
recomputes the preceding named moments/counters when following ancestry, so a
forged history cannot produce a self-consistent next state. This performs no
model or communication-graph replay.

Explicit imported-state optimization uses the shared scalar Torch optimizer on
CPU, with differentiable device copies for live derivative programs. The
artifact binds its implementation SHA, Torch/git/build, CPU architecture and
capability, Python/libc and subnormal behavior. A 4×2×8 probe found scalar Adam
CPU/CUDA rounding differences up to 7.45e-9, including moments; they were not
waived. CPU execution alone is not a universal ARM/x86 equivalence claim.
Numerical acceptance requires the bound runtime; a changed runtime needs actual
separate qualification before support. Training and adapter optimizer defaults
are unchanged. TypeScript codecs verify immutable origin, selected payload and
complete archive closure, not Torch arithmetic; the Python inference/gradient
consumer rechecks numeric execution before using optimized state.

Sampling-receipt-only artifacts use the same mixed native-artifact archive
closure as explicit writes and optimizers. A fresh typed HTTP caller that knows
only a server-side optimized input ID must archive its original producer,
gradient/update/program and optimizer ancestry before releasing the reply.

Repair qualification: 35 combined CPU cases and 14 Node store/archive cases
passed; CUDA passed 22 program/state cases. An additional six actual CUDA
two-update SGD/Adam attention/hybrid meta-gradient oracles passed against manual
execution through the same declared FP32-KV-cotangent scope (two Mellum hybrid
cases skip because that architecture has no convolution state). Its initial
Mellum failure was localized before the first update to a fixture comparing
storage cotangents against explicitly requested FP32 cotangents; the oracle
now binds identical arithmetic, with unchanged elementwise and relative-L2
gates. Live optimized state returns to the actor device once before graph use;
only the scalar optimizer's calculation is CPU-bound.

### RL archive continuation

An optimized state used by an RL actor is an explicit derived observation,
not a fresh producer capture or a new stochastic action. The exact bundle must
retain the same mixed state/tensor/program ancestry as the serving archive.
Crisp admission and objective plumbing reuse the shared dependency inventory
and numeric state verifier. Restore follows dependency order; captured external
states are checked from the immutable bundle before the shared packed forward.
Producer spans still use the existing one-pass replay mechanism, with no
recursive invocation rerun and no invented producer edge for an optimized ID.

`native_state_dependencies` is the shared inventory for store retention and RL
bundles. `NativeBundleStore` exposes verified immutable bundle bytes to the
existing state-derivation verifier. Both uniform and coefficient-policy
admission and actual policy-objective external-input loading use it; restored
mixed bundles load dependencies before state/gradient artifacts. Sampling
receipts must retain the actual derived input provenance, including learned
semantic decode observations.

Actual both-port tests capture two real sampled replies consuming a twice
optimized state, admit through production `native_stage.prepare`, obtain a
nonzero native model gradient from the registered RL objective, reverse bundle
entry order, restore an empty actor store, and reproduce the exact objective.
Missing history and altered capture provenance reject. Focused regression passed
20 cases, production coefficient/stage/policy integration passed 16 cases, and
the new two-model archive/objective test passed on actual CUDA. These are runtime
mechanism checks; fixture rewards do not claim a task oracle or model improvement.
