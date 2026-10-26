---
description: Prepare one mapParticipants evidence slice.
args: { goal: string; records: TransitionRequirement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Identify transport providers, equipment suppliers, caregivers, and receiving services with their actual responsibilities. Distinguish availability from acceptance, verify representation and information-sharing authority, and avoid assuming family relationships imply consent or capacity. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
