# Keep complete derivative references outside GPU memory

The production acceptance policy is unchanged: compare every declared parameter
and input derivative against exact cached execution, retaining the same gates
and preserving optimizer, parameters, RNG and existing gradients. Mellum's
current-parent check exhausted its 94.6 GiB GPU while retaining the complete
flat reference set beside cached backward and the restored optimizer. No SGD
occurred. The verifier's retention is the defect, not evidence against training.

1. Compute the same flat VJP with `autograd.grad`; copy each complete derivative
   to CPU without dtype conversion before constructing the cached forward.
2. As each cached derivative arrives, copy that exact tensor to CPU, compare
   the same full metrics with its CPU reference and release the reference.
   Observer pairs retain exact dtype/shape and use CPU storage, as the existing
   optimizer observer already does. Return the original hook value unchanged.
3. Keep all names, unused-gradient checks, absolute/relative gates and source
   preservation checks. Neither parameter gradients nor optimizer hooks run.
4. Report reference storage explicitly. Verify complete both-model derivatives,
   intentional wrong derivatives, CUDA reference-release lifetime, lazy cached
   construction and the existing whole-optimizer observer/transition checks.
5. Reuse only genuine completed actor/view/cache evidence under identical model
   arithmetic and loader code; the changed comparator must execute the complete
   pending gradient proof before production SGD.
