---
description: Prepare one interpretProposals evidence slice.
args: { goal: string; records: BudgetRound[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Translate informal proposals into clear intended outcomes, work scope, and affected groups. Preserve the proposer’s intent, distinguish a preferred implementation from the underlying benefit, and request clarification where ambiguity changes feasibility or eligibility. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
