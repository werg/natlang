---
description: Prepare one prepareBriefings evidence slice.
args: { goal: string; records: VolunteerTeam[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate role-specific briefings with task boundaries, practical logistics, and escalation routes. Share only necessary participant information, distinguish instructions from optional suggestions, and obtain acknowledgment for consequential responsibilities before deployment. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
