---
description: Prepare one planChannels evidence slice.
args: { goal: string; records: NarrativeInformation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Allocate information across captions, descriptions, transcripts, and supplementary explanations. Identify timing gaps and channel conflicts, prioritize essential meaning, and avoid duplicating dialogue in a way that obscures other necessary information. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
