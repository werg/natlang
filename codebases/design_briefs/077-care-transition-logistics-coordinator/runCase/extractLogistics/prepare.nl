---
description: Prepare one extractLogistics evidence slice.
args: { goal: string; records: CareTransition[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read the approved plan to identify logistical tasks and dependencies, preserving its wording for clinical requirements. Separate administrative coordination from treatment decisions and route any ambiguous clinical instruction back to the authorized team. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
