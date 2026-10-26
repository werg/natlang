---
description: Prepare one judgeIdentity evidence slice.
args: { goal: string; records: HistoricalCandidate[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare evidence for identity versus kinship, institutional succession, or coincidence. Weigh source reliability only within its actual scope, preserve competing candidates, and avoid treating a plausible story as proof of identity. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
