# Semantic Merge (`applications/sem-merge/`): Decomposition

Tier 2 flagship A6 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3): resolves 3-way merge conflicts while
preserving both intents, keeping non-overlapping merges crisp at zero model cost, and emitting explicit conflict
markers when intents genuinely clash.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `merge.ts`):**
  - `crispThreeWayChunks`: exact LCS-based 3-way line diff that merges all non-overlapping and identical edits
    deterministically with zero model calls, isolating only true overlapping conflicts into `ConflictHunk` regions.
  - Structured JSON branch: delegates conflicting JSON/config objects to `@natlang/node/std`'s `semanticMerge`.
  - Overlapping text/code branch: calls `decide(intentsCompatible, hunk)`, runs `mergeHunk.nl` inside
    `semanticRepair` verified by the repository's `verify(mergedContent, path)` callback, and emits explicit
    `<<<<<<< / ======= / >>>>>>>` conflict markers whenever intents are incompatible or verification fails.
- **Typed semantic policy (`.nl` leaves):**
  - `intentsCompatible.nl` (`readout: decision`, returns `boolean`): judges whether two overlapping edits pursue
    compatible intents.
  - `mergeHunk.nl` (returns `string`, guarded by `ensures:`): synthesizes a clean, syntactically valid merged block
    that preserves both intents without raw conflict markers.
