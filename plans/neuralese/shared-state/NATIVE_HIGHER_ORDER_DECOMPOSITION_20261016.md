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
