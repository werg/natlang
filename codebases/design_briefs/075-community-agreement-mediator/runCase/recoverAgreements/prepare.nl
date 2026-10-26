---
description: Prepare one recoverAgreements evidence slice.
args: { goal: string; records: CommunityDispute[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Reconstruct prior agreements from records and participant accounts, distinguishing signed or acknowledged terms from expectations and informal habits. Preserve disagreement about what was agreed and avoid treating one participant’s confident recollection as authoritative. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
