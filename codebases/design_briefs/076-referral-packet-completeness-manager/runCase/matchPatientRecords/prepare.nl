---
description: Prepare one matchPatientRecords evidence slice.
args: { goal: string; records: ReferralRequirement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Verify patient and episode identity using approved exact matching and reviewed evidence. Distinguish similar names, historical identifiers, and unrelated episodes; quarantine mismatches before content is included in a packet. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
