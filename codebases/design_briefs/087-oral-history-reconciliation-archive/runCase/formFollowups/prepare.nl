---
description: Prepare one formFollowups evidence slice.
args: { goal: string; records: RecollectionComparison[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate neutral follow-up questions that could clarify identities, chronology, or context without leading speakers toward a preferred account. Respect consent and participation boundaries, and distinguish clarification from challenging a person’s memory as false. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
