---
description: Prepare one interpretOffers evidence slice.
args: { goal: string; records: NegotiationRole[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Parse natural-language offers into issue-level terms, conditions, and commitment status. Distinguish exploration from a firm offer and qualified acceptance from final agreement; ask clarifying questions when ambiguity would change the negotiated outcome. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
