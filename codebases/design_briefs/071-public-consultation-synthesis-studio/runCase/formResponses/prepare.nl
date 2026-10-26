---
description: Prepare one formResponses evidence slice.
args: { goal: string; records: ConsultationTheme[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Draft evidence-linked response options for the accountable public body, including clarification, further investigation, or policy alternatives. Distinguish what the consultation establishes from questions requiring external evidence and avoid choosing political priorities not supplied by decision-makers. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
