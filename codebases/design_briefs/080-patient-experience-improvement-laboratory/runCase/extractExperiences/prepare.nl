---
description: Prepare one extractExperiences evidence slice.
args: { goal: string; records: ExperienceStudy[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Process surveys, complaints, and calls to identify administrative experiences and their context. Separate access or communication issues from clinical conclusions, preserve mixed experiences, and avoid inferring diagnoses or treatment quality from administrative feedback. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
