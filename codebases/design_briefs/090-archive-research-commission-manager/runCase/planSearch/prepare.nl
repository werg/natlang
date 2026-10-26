---
description: Prepare one planSearch evidence slice.
args: { goal: string; records: ResearchCommissionScope[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Select collections and search strategies using catalog structure, terminology variants, dates, and known access limits. Include negative search paths and record why a collection is relevant rather than relying only on keyword matches. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
