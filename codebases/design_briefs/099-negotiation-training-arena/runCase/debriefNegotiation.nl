---
description: debriefNegotiation domain stage for Negotiation Training Arena.
args:
  goal: string
  records: CounterpartAction[]
  evidence: EvidenceItem[]
returns: StageResultNegotiationDebrief
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftNegotiationDebrief values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Evaluate the observable trajectory and final agreement against learning goals, exact feasibility, and role constraints. Distinguish a good outcome from luck or prohibited conduct, explain missed trades using authorized debrief information, and propose targeted practice. The stage gate is: Feedback cites actual behavior and does not reward coercion or unsupported inference about motives.
