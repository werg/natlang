---
description: Prepare one recoverLearningMechanism evidence slice.
args: { goal: string; records: TrainingAdaptation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Analyze the original training to identify the skill exercised by each scenario and the elements that merely provide context. Separate essential decision structure from names, settings, idioms, or assumptions that may be adapted without changing the learning objective. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
