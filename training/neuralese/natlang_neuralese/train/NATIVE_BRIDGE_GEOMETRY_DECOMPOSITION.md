# Source03 bridge geometry reuse

## Purpose

The source03 fit bridge contains exact previously admitted auxiliary graphs for
complete source03 histories that are still pending as whole KEEP graphs. Its
already compiled forward/KV geometry is valid for the same LFM tokenizer, role
IDs, parser policy, limits, and exact graph rows. Reuse that evidence without
recompiling 1.29B token positions, while retaining the normal resident
continuation's full source, split, maximum-row, and admission checks.

## Natlang-native decomposition

1. **Resolve declared source.** Select the exact bridge corpus member and its
   staged source and row-binding sidecar. This is policy plumbing in the
   continuation caller; the evidence verifier does not choose a source.
2. **Authenticate the staged closure.** A crisp helper checks the bound corpus
   manifest, bridge source manifest, geometry supplement, and old-join
   provenance, then checks each referenced member's byte count and SHA-256 for the
   bridge source, sidecar, geometry supplement, old-join package, and their
   manifest/provenance files. It rejects missing members, path traversal,
   duplicate paths, or a changed receipt.
3. **Verify the old-to-bridge join.** A crisp verifier checks each bridge row
   against its source row binding, old admitted fit row, old coverage record,
   and old row-index entry. IDs, source graph and raw-row hashes, order, parser
   policy, tokenizer, role IDs, and forward/KV geometry must all agree. Each
   bridge ID must occur exactly once in the old fit and once in the bridge.
4. **Verify full appended-source coverage.** The shared geometry producer
   combines these bridge rows with ordinary/teacher parts in their declared
   order. No rows may be missing, duplicated, reordered, held, or pending. The
   normal row limit and physical maximum-row proof remains active.
5. **Publish immutable geometry evidence.** The existing locked atomic writer
   binds the resulting source entries, tokenizer/roles, index witnesses,
   maxima, and every bridge/ordinary evidence receipt. It does not change
   source manifests, recipes, or exposure.

Only exact file durability, JSON parsing, hashes, path containment, row joins,
geometry equality, and atomic publication are crisp. Source selection and
whether a bridge belongs in the next recipe continuation remain explicit
caller decisions. Existing ordinary and teacher geometry paths retain their
current schemas and behavior.

## Validation

Focused tests must accept one real-source-shaped bridge fixture and reject
source/sidecar/receipt tampering, a duplicate or missing bridge row, wrong
source-row binding, wrong old-fit geometry, wrong parser/tokenizer/roles,
changed order, and a forged maximum. The actual source03 staged closure should
also pass without invoking a whole-graph geometry compile.
