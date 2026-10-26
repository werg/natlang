---
description: Prepare one frameCommission evidence slice.
args: { goal: string; records: ArchiveCommission[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Interpret the requester’s goal and decompose it into answerable historical questions. Identify what available collections could establish, distinguish factual lookup from interpretive synthesis, and agree a bounded effort and stopping rule before searching indefinitely. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
