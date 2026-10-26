---
description: Prepare one assembleDossier evidence slice.
args: { goal: string; records: ArchiveFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Produce an evidence dossier with findings, source references, search history, unresolved leads, and reuse restrictions. Tailor the narrative to the commission while preserving methodological limits and separating the researcher’s interpretation from quoted or recorded facts. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
