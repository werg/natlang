---
description: Prepare one interpretProfiles evidence slice.
args: { goal: string; records: MissionRequirement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read volunteer profiles and confirmations to distinguish self-reported skills, verified qualifications, availability, and accepted roles. Preserve limitations and preferences, and avoid treating past participation as consent to a new assignment. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
