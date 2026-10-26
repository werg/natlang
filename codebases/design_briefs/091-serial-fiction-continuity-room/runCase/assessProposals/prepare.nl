---
description: Prepare one assessProposals evidence slice.
args: { goal: string; records: StoryObligation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Evaluate proposed twists or scenes against chronology, knowledge, motivations, and world rules. Separate genuine contradiction from a reveal that reinterprets prior evidence, and identify what earlier material would need revision to support the new direction. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
