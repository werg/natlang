# Exact report hashing without delaying training continuation

Training supervision continues from the saved update as soon as its exact
checkpoint and terminal report are durable. Large per-source inventories are
checkpoint-backed evidence, not a reason to leave an accelerator idle.

1. Preserve every existing report field and checkpoint detail reference.
2. Hash the identical canonical JSON bytes for the detail being referenced.
3. Stream the outer inventory in batches of at most 1,024 entries. Encode each batch with the
   existing Python JSON encoder's native implementation, retaining sorted keys,
   Unicode, float formatting, separators and JSON key conversion.
4. Retain the normal encoder for scalar details. Do not change training,
   optimizer, exposure, admission or numerical policy.

The crisp helper is exact serialization and SHA-256 plumbing. It replaces the
Python encoder's per-scalar generator on large flat inventories, with temporary
storage bounded by one batch plus the sorted outer key list, rather than the entire
encoded inventory. A nested entry still needs its own encoded representation. Existing end-to-end
trainer/detail-resolution checks and canonical-byte equivalence checks cover
this boundary; an operational benchmark reports speed without a timing gate.

Validation: 22 report tests passed, including actual trainer checkpoint/detail
resolution and canonical equivalence across multiple batch boundaries. On this
DGX, 200,000 synthetic source-group records hashed in 0.162 seconds versus 0.443
seconds for the previous scalar generator (2.74x), with identical SHA-256.
Receipt: `/home/werg/data/dgxops/operation-report-hash-benchmark-20261017.json`.
This measures serialization only, not training throughput or the duration of a
complete production report.
