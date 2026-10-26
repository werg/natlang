---
description: Prepare one identifyPathways evidence slice.
args: { goal: string; records: ServiceSituation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare known facts with supplied service descriptions and conditions to identify potentially relevant routes. Preserve multiple plausible pathways, explain uncertainty, and avoid presenting a preliminary match as an official eligibility decision. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
