# Semantic Merge

Tier 2 Semantic-FP flagship application (`A6`) for intent-preserving three-way file merging.

## Highlights

- **Crisp by construction**: Non-overlapping line hunks and identical edits merge deterministically in TypeScript with 0 model calls.
- **Structured JSON merging (`semanticMerge`)**: Conflicting configuration/JSON objects merge key-by-key by intent.
- **Verified code conflict resolution (`semanticRepair` + `ensures:`)**: Overlapping compatible hunks are judged by `intentsCompatible.nl` (`readout: decision`), combined by `mergeHunk.nl`, and repaired against the repository's exact verifier callback.
- **No silent drops**: Incompatible intents or unverified repairs always emit explicit `<<<<<<< / ======= / >>>>>>>` conflict markers.
