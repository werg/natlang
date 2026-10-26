---
description: Prepare one prepareRoles evidence slice.
args: { goal: string; records: NegotiationScenario[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate role materials that provide enough information to negotiate coherently while preserving intended asymmetry. Distinguish private preferences from facts that may be disclosed, and bind each participant’s authority and conduct limits explicitly. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
