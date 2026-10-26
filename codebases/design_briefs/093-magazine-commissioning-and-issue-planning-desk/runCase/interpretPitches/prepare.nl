---
description: Prepare one interpretPitches evidence slice.
args: { goal: string; records: MagazineIssue[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read pitches against the issue brief to identify the actual angle, reporting plan, and audience value. Distinguish an interesting topic from a viable story, preserve uncertainty about access or evidence, and identify overlap with already commissioned work. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
