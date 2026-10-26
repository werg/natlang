---
description: Prepare one followEvidence evidence slice.
args: { goal: string; records: ArchiveSearchPlan[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read retrieved records in context, extract relevant evidence, and formulate bounded follow-up leads. Distinguish direct evidence from references to missing sources, avoid circular citation chains, and stop low-value leads under the agreed effort budget. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
