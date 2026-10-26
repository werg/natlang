---
description: Prepare one verifyEdition evidence slice.
args: { goal: string; records: StoryRevision[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Re-run continuity checks over affected scenes and downstream knowledge states, inspect arc payoffs, and assemble a coherent editorial diff. Record approved canon changes and preserve prior editions so future work uses the correct story-world state. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
