---
description: Prepare one generateAlternatives evidence slice.
args: { goal: string; records: ContextFitFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Invent alternative settings, examples, and interactions that remain plausible for the supplied context. Preserve the same relevant ambiguity, constraints, and skill demand, and avoid making the adapted task easier or changing the intended lesson unintentionally. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
