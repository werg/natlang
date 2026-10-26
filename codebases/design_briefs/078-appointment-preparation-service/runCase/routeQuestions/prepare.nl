---
description: Prepare one routeQuestions evidence slice.
args: { goal: string; records: PreparationChecklist[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Form focused questions for the appropriate staff when instructions conflict with practical circumstances or remain unclear. Track authoritative responses, avoid answering medical questions from general model knowledge, and update the checklist only from approved clarification. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
