---
description: Prepare one interpretAuthorization evidence slice.
args: { goal: string; records: HealthRecordRequest[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read the request and authorization together to determine recipient, purpose, categories, dates, and any restrictions. Distinguish a requested scope from an authorized scope and route ambiguous authority to the records officer before retrieval. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
