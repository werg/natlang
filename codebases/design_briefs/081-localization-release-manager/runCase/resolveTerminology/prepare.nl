---
description: Prepare one resolveTerminology evidence slice.
args: { goal: string; records: LocalizationSegment[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Match terms to concepts and approved locale guidance, considering polysemy, grammatical role, and product meaning. Preserve scoped exceptions and route genuine terminology conflicts to the owner rather than choosing whichever translation appears most often. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
