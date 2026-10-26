---
description: Prepare one interpretCircumstances evidence slice.
args: { goal: string; records: PreparationRequirement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read the patient’s stated practical circumstances and communication preferences without inferring diagnoses or capability. Identify logistical barriers and questions requiring staff attention, and verify authority before involving a representative. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
