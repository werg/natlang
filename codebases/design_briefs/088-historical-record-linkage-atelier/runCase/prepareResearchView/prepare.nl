---
description: Prepare one prepareResearchView evidence slice.
args: { goal: string; records: HistoricalNetwork[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Publish a reviewable linkage view with source-level explanations and explicit alternatives. Support corrections and reversible splits, preserve stable source identifiers, and expose uncertainty to downstream researchers instead of exporting only a deceptively clean graph. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
