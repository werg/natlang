---
description: Prepare one generateLinks evidence slice.
args: { goal: string; records: HistoricalObservation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate candidate links using names, chronology, places, relationships, and contextual descriptions. Include plausible aliases while preserving hard incompatibilities, and audit records excluded by blocking so unusual spellings do not disappear from consideration. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
