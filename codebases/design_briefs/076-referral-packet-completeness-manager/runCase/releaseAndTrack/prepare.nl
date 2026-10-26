---
description: Prepare one releaseAndTrack evidence slice.
args: { goal: string; records: ReferralRequest[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Assemble a recipient-bound packet, verify exact patient and service identifiers, and obtain required review before transmission. Track delivery, receipt, and administrative acceptance separately, preserving rejection reasons and follow-up obligations. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
