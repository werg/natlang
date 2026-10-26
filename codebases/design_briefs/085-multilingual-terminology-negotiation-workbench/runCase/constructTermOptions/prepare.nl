---
description: Prepare one constructTermOptions evidence slice.
args: { goal: string; records: MultilingualConcept[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Propose terms and scoped usage rules considering approved domain conventions, grammatical forms, and audience needs. Explain tradeoffs among clarity, continuity, and precision, and retain regional or contextual variants where one universal term would change meaning. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
