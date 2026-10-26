---
description: Prepare one reconcileFindings evidence slice.
args: { goal: string; records: ArchiveLead[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare evidence across sources while preserving chronology, attribution, and disagreement. State what was found and what remains uncertain, and do not turn an unsuccessful search in limited collections into proof that an event never occurred. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
