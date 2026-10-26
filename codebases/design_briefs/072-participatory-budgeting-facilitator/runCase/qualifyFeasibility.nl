---
description: qualifyFeasibility domain stage for Participatory Budgeting Facilitator.
args:
  goal: string
  records: ProposalRelationship[]
  evidence: EvidenceItem[]
returns: StageResultProposalFeasibility
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftProposalFeasibility values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare proposals with supplied eligibility, resource, and delivery rules, using exact estimates and qualified assessments where necessary. Separate infeasibility from missing information and provide a clear path for revision rather than silently excluding difficult proposals. The stage gate is: Eligibility and feasibility decisions cite supplied rules and verified estimates.
