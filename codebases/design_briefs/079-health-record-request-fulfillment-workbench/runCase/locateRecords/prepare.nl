---
description: Prepare one locateRecords evidence slice.
args: { goal: string; records: RecordAuthorization[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Search authorized inventories using verified patient and episode identifiers, preserving date and category boundaries. Distinguish unavailable records from records outside scope and prevent similar identities or merged historical accounts from contaminating the packet. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
