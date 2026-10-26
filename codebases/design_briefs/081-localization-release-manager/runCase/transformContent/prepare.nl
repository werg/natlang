---
description: Prepare one transformContent evidence slice.
args: { goal: string; records: LocaleTermDecision[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Translate or adapt according to communicative purpose while preserving product facts, obligations, and constraints. Generate alternatives for culturally or linguistically sensitive phrasing using supplied guidance, and keep unresolved meaning questions visible rather than hiding them in fluent text. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
