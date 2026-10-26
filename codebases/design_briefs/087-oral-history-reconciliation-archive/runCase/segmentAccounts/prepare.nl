---
description: Prepare one segmentAccounts evidence slice.
args: { goal: string; records: OralHistoryProject[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Segment interviews into attributable accounts while preserving surrounding context and the speaker’s expressions of uncertainty. Distinguish firsthand recollection, hearsay, later interpretation, and quoted speech rather than presenting them all as equivalent historical evidence. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
