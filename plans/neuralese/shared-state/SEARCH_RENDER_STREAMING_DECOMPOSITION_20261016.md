# Search/OWL renderer bounded streaming decomposition

## Natlang-native policy and exact mechanism

The rendering policy remains the existing selected-record, split-protection,
source-group decontamination, teacher-message rendering, whole-trajectory
coverage, and duplicate-document policy. This change does not select, reject,
truncate, or relabel any source row. It changes only how the already-selected
records are materialized and durably written.

1. The existing deterministic selector continues to produce all selected
   source rows and its source-bound `selection-state.json` checkpoint.
2. A renderer worker streams one selected shard into a bounded disk-backed
   record index, keyed by exact source chain and split. SQLite is the crisp
   storage mechanism because exact resumable random lookup is an external
   durability requirement; source JSON and all original fields remain intact.
3. The worker visits chains in deterministic first-source order, loads one
   chain's complete records, applies the existing hold/decontamination and
   authenticated writer-context rules, and emits the same full/header/whole
   rows. A chain is the smallest unit that preserves maximal-trajectory
   membership and the existing causal source joins.
4. Each completed chain's output rows and omission rows are appended as an
   independently checksummed compressed frame. Only after all output frames
   are closed, fsynced, and bound in the progress receipt does the chain
   frontier advance. A restart verifies frame hashes, truncates any uncommitted
   tails, and continues at the next chain.
5. Existing global merge remains responsible for cross-chain source-group
   closure and exact duplicate-document merging. Its inputs are consumed in the
   original shard/chain order; the resulting documents, omissions, role masks,
   provenance, and counts must match the existing whole-shard renderer.

## Why crisp code is used

Streaming JSON decode, SQLite indexing, compressed frame writing, fsync, and
checksums are precise data and durability plumbing. The actual rendering and
all decisions continue to call the existing shared tokenizer, source
authentication, decontamination, and native chat renderer. No model-facing
prompt or policy is added.

## Qualification

Tests compare the bounded implementation against the retained legacy renderer
on representative multi-chain train/test fixtures for all requested modes,
including duplicate documents, masked targets, protected groups, and source
context. An interrupted run must recover from committed frames and produce
byte-identical final JSONL and receipt counts to an uninterrupted run. A real
selected Search shard is then the bounded operational validation; its exact
selection checkpoint and source bindings stay unchanged.
