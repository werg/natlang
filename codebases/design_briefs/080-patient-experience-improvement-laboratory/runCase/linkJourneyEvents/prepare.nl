---
description: Prepare one linkJourneyEvents evidence slice.
args: { goal: string; records: ExperienceEpisode[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Relate feedback to appointment, contact, and referral events using authorized identifiers and time windows. Distinguish repeated reports of one episode from independent experiences, and preserve uncertain linkage rather than exposing unrelated patient records. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
