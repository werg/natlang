---
description: Prepare one chooseQuestions evidence slice.
args: { goal: string; records: ServicePathway[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Select the next question by how much it resolves relevant pathway uncertainty while minimizing burden and sensitivity. Reuse authorized existing evidence, explain why the information is needed, and allow the person to decline or report that it is unknown. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
