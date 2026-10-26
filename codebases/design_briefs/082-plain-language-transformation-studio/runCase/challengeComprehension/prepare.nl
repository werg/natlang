---
description: Prepare one challengeComprehension evidence slice.
args: { goal: string; records: PlainLanguageDraft[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Create realistic comprehension questions and boundary scenarios that test whether readers would act correctly. Include cases involving exceptions and negative requirements, then compare source and rewrite answers to detect semantic drift. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
