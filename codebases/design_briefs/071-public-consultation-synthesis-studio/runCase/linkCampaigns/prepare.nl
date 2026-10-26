---
description: Prepare one linkCampaigns evidence slice.
args: { goal: string; records: ConsultationArgument[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Detect copied or campaign-template language while preserving independently added reasons and legitimate participation counts. Distinguish duplicate ingestion from multiple people submitting the same position, and report volume and argument diversity separately. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
