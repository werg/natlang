---
description: Prepare one generateDishes evidence slice.
args: { goal: string; records: MenuRequirement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Invent distinct dishes using available ingredients and approved culinary procedures, explaining flavor, texture, and service intent. Propose substitutions only when their properties and dietary implications are verified, and preserve specialist food-safety review where needed. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
