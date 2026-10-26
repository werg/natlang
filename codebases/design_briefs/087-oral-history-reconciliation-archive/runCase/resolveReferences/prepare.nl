---
description: Prepare one resolveReferences evidence slice.
args: { goal: string; records: OralHistoryAccount[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Resolve names, places, and relationships using authorized contextual evidence, preserving aliases and uncertainty. Avoid merging people with similar names or assuming a pronoun’s referent when the interview context supports multiple interpretations. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
