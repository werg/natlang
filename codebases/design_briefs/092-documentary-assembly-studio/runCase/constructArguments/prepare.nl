---
description: Prepare one constructArguments evidence slice.
args: { goal: string; records: DocumentarySegment[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Develop alternative narrative structures around the editorial question and available evidence. Include relevant contradictions and uncertainty, distinguish thematic juxtaposition from causal implication, and avoid inventing connective facts to make the story more dramatic. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
