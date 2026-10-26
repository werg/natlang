---
description: Prepare one writeArtifacts evidence slice.
args: { goal: string; records: PuzzleNode[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Create in-world documents, dialogue, labels, and clues that support the intended reasoning without relying on designer-only knowledge. Generate plausible player misreadings, preserve thematic coherence, and distinguish fair misdirection from ambiguity that makes the puzzle underdetermined. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
