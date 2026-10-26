---
description: Prepare one interpretMission evidence slice.
args: { goal: string; records: VolunteerMission[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Translate the mission brief into concrete tasks, capability requirements, and practical constraints. Distinguish essential qualifications from preferences, identify supervision needs, and preserve safety or safeguarding requirements supplied by the organization. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
