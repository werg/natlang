---
description: Prepare one interpretRequirements evidence slice.
args: { goal: string; records: ReferralCase[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read the receiving service’s supplied administrative requirements and the clinician’s referral instructions. Identify required records, forms, dates, and signatures while preserving clinician-provided priority and avoiding any independent judgment about clinical necessity. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
