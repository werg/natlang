---
description: Prepare one reviewProduction evidence slice.
args: { goal: string; records: DocumentaryEdit[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Review the proposed assembly for factual support, rights, attribution, accessibility, and unresolved editorial issues. Distinguish verified claims from attributed testimony and commission targeted fact-checking or clearance tasks before locking the edit. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
