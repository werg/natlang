---
description: Prepare one assessCompleteness evidence slice.
args: { goal: string; records: ReferralRecordMatch[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare each requirement with submitted documents, checking scope, dates, signatures, and readability. Distinguish document presence from actual required content and route clinical-content questions to authorized staff rather than interpreting them autonomously. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
