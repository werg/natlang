# Transfer exact model views using an existing verified byte basis

The owner requires prompt actual model-quality feedback and recoverable progress.
DGX's current view transfer is constrained to roughly3MB/s even after moving its
partial download to NVMe. The interface receives roughly6MB/s total. A bounded
32MiB sample of two existing Mellum views reconstructed byte-for-byte and
compressed3.4x as XOR differences, versus1.28x for direct compression. This is
sampling evidence, not a full-artifact compression or transport qualification.

1. Keep independently verified full training recovery and original inference
   views as authorities. This transport does not alter model arithmetic, weights,
   optimizer/RNG/exposure, Neuralese semantics or any task/quality requirement.
2. Identify target and an available immutable basis by exact SHA-256 and bytes.
   Encode streaming XOR bytes through zstd, authenticating both complete inputs
   and stable source file identities before publishing an immutable container.
3. Bind the private container, target and basis explicitly. A decoder validates
   the manifest, both complete input identities and the entire reconstructed
   target before exposing a file. Never treat a copied partial as a valid model.
4. Reuse existing private export, transfer admission, partial-download and queue
   mechanisms. Keep normal raw transfer when a qualified basis/container is
   unavailable. Preserve actual temporary-byte accounting and free-space guards.
5. Validate full real-view reconstruction and private transfer before deployment.
   Retain each actually consumed basis until successor bytes are independently
   verified; bind those dependencies in retention. Resume interrupted work
   through durable handles rather than restarting after observation deadlines.
6. Apply the same mechanism to both model lines. Actual task quality is evaluated
   on the restored original artifact, not on a quantized approximation.

These are exact byte, storage, custody and durability helpers. The natural-
language policy above chooses useful available verified transport. This is not
Neuralese semantic adaptation or the later shared-space VAE.
