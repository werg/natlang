---
description: Prepare one replayLearning evidence slice.
args: { goal: string; records: AdaptedScenario[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare original and adapted scenarios through representative correct, incorrect, and borderline responses. Detect shifts in difficulty or meaning, inspect whether examples imply unsupported norms, and revise only the elements responsible for the drift. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
