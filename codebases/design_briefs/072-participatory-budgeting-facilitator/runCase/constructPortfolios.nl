---
description: constructPortfolios domain stage for Participatory Budgeting Facilitator.
args:
  goal: string
  records: ProposalFeasibility[]
  evidence: EvidenceItem[]
returns: StageResultCivicPortfolio
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCivicPortfolio values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Enumerate feasible combinations under exact budget and dependency constraints, accounting for shared work and mutually exclusive uses. Present alternatives for deliberation without autonomously choosing political priorities or inferring participant preferences from personal data. The stage gate is: Exact budget feasibility and complete disclosure of exclusions and dependencies.
