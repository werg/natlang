---
description: Prepare one requestMaterial evidence slice.
args: { goal: string; records: ReferralGap[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Group missing-material requests by authorized source, ask only for necessary records, and preserve patient consent and permitted disclosure scope. Avoid repeated requests for existing evidence and track who is responsible for each unresolved gap. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
