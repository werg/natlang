---
description: Decide field-level findings for writeArtifacts.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Create in-world documents, dialogue, labels, and clues that support the intended reasoning without relying on designer-only knowledge. Generate plausible player misreadings, preserve thematic coherence, and distinguish fair misdirection from ambiguity that makes the puzzle underdetermined. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
