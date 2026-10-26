---
description: prepareDeliberation domain stage for Participatory Budgeting Facilitator.
args:
  goal: string
  records: CivicPortfolio[]
  evidence: EvidenceItem[]
returns: StageResultBudgetDecisionPacket
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftBudgetDecisionPacket values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Produce accessible comparisons, maps or structured location views, and clear decision procedures. Preserve proposer corrections, record the authorized community decision, and transfer selected projects into accountable delivery plans with explicit acceptance criteria. The stage gate is: Selection follows the published process and the selected portfolio matches its approved scope.
