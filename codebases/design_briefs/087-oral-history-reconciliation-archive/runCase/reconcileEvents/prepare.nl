---
description: Prepare one reconcileEvents evidence slice.
args: { goal: string; records: OralHistoryReference[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare accounts of potentially shared events, aligning chronology and scope while preserving different perspectives. Distinguish factual contradiction from differences in emphasis or memory precision, and do not manufacture one authoritative narrative from majority agreement. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
