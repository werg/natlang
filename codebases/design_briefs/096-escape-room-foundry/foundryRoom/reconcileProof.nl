---
description: Compare a supplied puzzle proof with an independently recomputed graph result for the exact room revision.
args: { computed: PuzzleVerification; supplied: PuzzleProof }
returns: PuzzleProofCheck
---
Bind comparison = `comparePuzzleProofs(computed, supplied)`. Preserve every exact differing field. A supplied proof is corroborating evidence only; use the computed graph result as the operative proof. Any revision or field mismatch blocks the room packet and must not be repaired by copying the supplied values into the computed proof.
