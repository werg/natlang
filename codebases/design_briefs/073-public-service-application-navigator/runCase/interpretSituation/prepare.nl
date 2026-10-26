---
description: Prepare one interpretSituation evidence slice.
args: { goal: string; records: ServiceNavigation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read the person’s stated situation and supplied documents to identify needs and established facts. Avoid inferring sensitive characteristics, distinguish missing information from ineligibility, and route urgent concerns according to approved service procedures. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
