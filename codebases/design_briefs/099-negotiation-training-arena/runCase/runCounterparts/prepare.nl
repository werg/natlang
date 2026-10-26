---
description: Prepare one runCounterparts evidence slice.
args: { goal: string; records: TrainingOffer[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Choose simulated counterpart responses from permitted actions using only that role’s information and preferences. Adapt strategy to the interaction without violating conduct rules, inventing authority, or exposing hidden utility values to the learner. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
