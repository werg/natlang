---
description: Prepare one interpretIntent evidence slice.
args: { goal: string; records: GameWorld[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Interpret player language in the visible world context and map it to permitted action types. Distinguish speech, intention, and executed action, ask only consequential clarifications, and avoid granting capabilities because a player narrates success confidently. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
