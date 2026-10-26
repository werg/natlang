---
description: Prepare one negotiateDecision evidence slice.
args: { goal: string; records: TermOption[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Present alternatives and concrete usage examples to the relevant experts, reconcile their comments, and distinguish stylistic preference from conceptual disagreement. Record scoped acceptance and unresolved dissent rather than treating a majority vote as proof of semantic equivalence. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
