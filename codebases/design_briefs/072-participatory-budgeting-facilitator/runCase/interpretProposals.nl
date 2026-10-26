---
description: interpretProposals domain stage for Participatory Budgeting Facilitator.
args:
  goal: string
  records: BudgetRound[]
  evidence: EvidenceItem[]
returns: StageResultCivicProposal
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCivicProposal values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Translate informal proposals into clear intended outcomes, work scope, and affected groups. Preserve the proposer’s intent, distinguish a preferred implementation from the underlying benefit, and request clarification where ambiguity changes feasibility or eligibility. The stage gate is: Proposal normalization does not silently change the project’s purpose.
