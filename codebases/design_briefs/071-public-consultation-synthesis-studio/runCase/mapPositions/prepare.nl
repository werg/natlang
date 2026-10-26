---
description: Prepare one mapPositions evidence slice.
args: { goal: string; records: SubmissionRelation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Group arguments by the actual concern and reasoning, not only topic words. Preserve minority and cross-cutting positions, identify disagreements about facts versus values, and represent tradeoffs without averaging incompatible views into false consensus. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
