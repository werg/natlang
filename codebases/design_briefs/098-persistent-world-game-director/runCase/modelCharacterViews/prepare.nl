---
description: Prepare one modelCharacterViews evidence slice.
args: { goal: string; records: PlayerIntent[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Build each character’s view from witnessed events, received messages, memory, and approved world facts. Separate true facts from beliefs and secrets, preserve obligations and motivations, and prevent the narrator’s global knowledge from leaking into character decisions. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
