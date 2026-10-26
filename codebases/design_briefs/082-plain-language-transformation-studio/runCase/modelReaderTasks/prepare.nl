---
description: Prepare one modelReaderTasks evidence slice.
args: { goal: string; records: MeaningUnit[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Interpret what the intended reader must understand or do using supplied audience guidance. Identify decision points and necessary background without stereotyping ability, and distinguish information that can be reorganized from information that cannot be omitted. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
