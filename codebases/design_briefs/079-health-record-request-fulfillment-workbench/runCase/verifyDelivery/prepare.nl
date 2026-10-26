---
description: Prepare one verifyDelivery evidence slice.
args: { goal: string; records: RecordFulfillment[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Recheck authorization immediately before recipient-bound delivery, record actual receipt where available, and track partial fulfillment or corrections. Preserve an audit trail without exposing sensitive record bodies in operational logs or notifications. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
