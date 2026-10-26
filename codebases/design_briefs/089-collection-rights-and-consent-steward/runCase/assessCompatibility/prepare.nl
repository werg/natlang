---
description: Prepare one assessCompatibility evidence slice.
args: { goal: string; records: ProposedCollectionUse[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare each proposed act with supplied rights and consent conditions, including interactions among multiple rightsholders. Preserve scoped exceptions and refer ambiguous legal interpretation to the rights officer rather than selecting the most permissive reading. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
