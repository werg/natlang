---
description: Prepare one generateRepairs evidence slice.
args: { goal: string; records: StoryProposalFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate distinct repairs such as changing the new scene, seeding earlier evidence, or revising an approved premise through explicit editorial choice. Preserve voice and emotional intent, and explain narrative tradeoffs rather than selecting the easiest continuity patch. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
