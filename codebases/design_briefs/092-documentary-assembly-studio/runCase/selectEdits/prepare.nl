---
description: Prepare one selectEdits evidence slice.
args: { goal: string; records: DocumentaryStructure[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Assemble candidate sequences and transitions under exact duration constraints. Inspect how cuts, adjacency, and narration change the apparent meaning of statements, preserving necessary context and identifying misleading implications even when every individual clip is authentic. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
