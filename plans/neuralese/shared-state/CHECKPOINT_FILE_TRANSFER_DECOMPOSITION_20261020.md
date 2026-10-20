# Exact file-backed checkpoint transfers

The existing recipe chooses checkpoint input offload and the checkpoint wrapper
chooses whether host transfers use pinned memory. These remain the policy
interfaces; no model-facing instruction or training objective changes.

1. Keep the native nonreentrant checkpoint and its saved-tensor hooks.
2. Preserve each saved tensor's dtype, physical span, strides and device.
3. When pinned transfers are requested for CUDA tensors, move exact chunks
   through one bounded, reusable pinned host buffer. Finish each transfer before
   reusing the buffer or exposing its file-backed destination.
4. Flush file-backed values and release their page cache as before. Restore
   independent device storage, including repeated reads and higher derivatives.
5. Keep the existing pageable implementation for CPU tensors and explicitly
   disabled pinning. Report actual staging memory and transfer counts.

These are crisp memory/storage operations. The buffer is bounded because
production needs file storage to avoid accumulating anonymous host activations.
The Mellum R6 real-update profile measured 15.96 seconds of pageable DtoH and
13.38 seconds of pageable HtoD transfers in a 75.52-second four-update window.
The file implementation currently ignores the wrapper's pin-memory selection.
Fix that mechanism and verify exact native graph gradients, optimizer updates,
RNG, layouts and holder lifetimes. Target-device throughput and whole-recipe
capacity still require measurement before deploying the new runtime.
