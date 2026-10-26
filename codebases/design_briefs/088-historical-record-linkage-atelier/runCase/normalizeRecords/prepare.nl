---
description: Prepare one normalizeRecords evidence slice.
args: { goal: string; records: HistoricalLinkage[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Preserve original spellings, date expressions, and geographic descriptions while creating normalized search forms. Distinguish transcription uncertainty from historical variation, and avoid converting approximate dates or changing place names into falsely precise modern facts. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
