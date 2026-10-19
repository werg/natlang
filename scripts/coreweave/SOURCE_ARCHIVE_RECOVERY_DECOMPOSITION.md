# Stable private source recovery authority

Policy: the same authenticated private archive and restored member inventory keep
their original durable source receipt. A later verification is an observation,
not a new source identity. Different private custody, context or bytes require an
explicit continuation and cannot overwrite that receipt.

1. Hold one receipt lock across the existing full archive verification and atomic
   per-member restoration. Preserve the original archive's source decisions.
2. Compare the proposed private/context authority against the existing receipt
   before any restore. Reject a changed authority or symlink receipt.
3. Run the existing restore, including decoded archive and member SHA/size checks.
4. Publish the receipt once, using exact atomic durability. On reuse compare every
   field except the observation timestamp and retain the original bytes/inode.
5. Append a separately fsynced fresh observation containing verification time and
   the exact sealed receipt SHA/size. Normal source/admission verification still
   authenticates the private receipt and all original member bindings.

This is crisp outside-world archive verification, serialization and durability;
it makes no quality, source-selection or training-admission decision.
