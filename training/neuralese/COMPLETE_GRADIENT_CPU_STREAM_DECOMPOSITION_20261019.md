# Complete-gradient comparison CPU streaming (2026-10-19)

The transition policy remains exact: compare every declared trainable gradient
and the explicit input-embedding gradient against the existing reference, with
the same dtype, tolerances, optimizer state, and parameter-gradient preservation.
The current transition obtains an entire candidate `autograd.grad` tuple on the
GPU before moving any member to CPU. The tuple is comparison data, not training
state, and need not remain resident as one model-sized allocation.

The comparison stage will run one ordinary backward traversal and collect each
completed leaf gradient immediately after autograd accumulates it. The flat
reference keeps its CPU gradient map. For the cached traversal, each
post-accumulate hook copies a single completed gradient to CPU, compares it
against its matching reference, and releases it before the next leaf completes;
it never builds a second model-sized CPU gradient map. A hook on the explicit
input tensor does the same CPU copy for its gradient while returning the
original device gradient so autograd semantics remain intact. Existing VJP
parameter hooks run before post-accumulate hooks, so their exact carrier audits
and one-shot device accumulator release remain authoritative.

Exact code is required for leaf-gradient collection, hook installation/removal,
temporary parameter and input `.grad` isolation/restoration, and proof accounting. The comparison
policy remains the existing exact complete-inventory and numeric gates. This
does not change parameter values, optimizer updates, gradient tolerances, or
runtime arithmetic. Trainable targets must be distinct leaves; unsupported
non-leaf targets fail closed rather than being silently omitted. Unused targets
remain represented as `None` and are compared under the existing unused-gradient
rule.

Validation compares the streamed path to `autograd.grad` on tiny CPU and CUDA
graphs, including BF16 carrier fan-in, an unused leaf, preexisting parameter
and input `.grad` objects/values (leaf and retained non-leaf inputs), explicit
input gradients, backward exception cleanup, and a real cached graph transition
fixture. Cached CPU gradients are verified to reach the observer one at a time.
CUDA peak evidence must report the complete
transition peak, not just post-backward allocations. No live runtime is changed
by this work.
