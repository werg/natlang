# Acquired admission resume decomposition (2026-10-17)

## Purpose

Paired native admission can validate a complete source export for one model,
then be preempted while validating the other model. The current all-or-nothing
audit discards that completed computation on restart. Persist completed
50-session shard validations and final per-model results, then reuse them only
when the exact source range, shard files, verifier code, model/tokenizer
inventory, source closure, and current protection/decision holds still match.

## Natlang-native decomposition

1. **Choose whether a saved model proof still applies** — natlang policy over
   the exact source member, model view, current split/quality/protection
   evidence, and conversion receipt. This remains within the existing admission
   policy; a cache hit never creates admission or training credit.
2. **Compute exact byte identities** — crisp helper using the shared
   `sha256_file_hex_cached(..., trust_archive_receipt=False)` path. Kernel
   device/inode/size/mtime/ctime guard and the shared hashing helper's
   before/after checks make unchanged-byte reuse safe under the repository's
   stated non-adversarial filesystem model. Content reads occur on cache misses.
3. **Validate one complete source shard** — existing source and graph
   verifiers check its exact 50-row interval, omissions, source identity,
   target rows, held splits, and all whole and auxiliary reasoning views.
4. **Durably checkpoint the completed proof** — crisp atomic immutable write
   of each fully validated shard's graph proofs, groups, counts, and exact
   cache identity. A partial cache file is never readable as a completed proof.
5. **Merge the complete model** — recheck cross-shard graph/source uniqueness,
   full view coverage, accounting, split separation, and exact aggregate bytes
   as the ordered concatenation of the shard outputs. A final per-model cache
   is written only after those checks pass.
6. **Recheck current policy before final publication** — existing current
   protection, decision-hold, full-source-closure, and source-parity checks run
   on every invocation and again at the final boundary. Any changed input
   yields a cache miss or fails closed; old cached eligibility is never
   restored.

The cache is operational scratch outside the published admission output. Its
contents are not registered, do not add a source, and do not imply training
exposure. The immutable final audit remains the sole output consumed by corpus
registration.
