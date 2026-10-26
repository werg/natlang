---
description: Prepare one assessContextFit evidence slice.
args: { goal: string; records: TrainingMechanism[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare scenario elements with supplied audience guidance and reviewer input. Identify unfamiliar references, inappropriate assumptions, or practical mismatches without inferring cultural traits from nationality, ethnicity, or other sensitive attributes. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
