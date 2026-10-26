---
description: Prepare one interpretUse evidence slice.
args: { goal: string; records: CollectionPermission[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Translate the proposed exhibition, publication, or derivative use into concrete acts and distribution scope. Identify cropping, translation, online access, commercial use, and duration where relevant, rather than treating all publication as one undifferentiated permission. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
