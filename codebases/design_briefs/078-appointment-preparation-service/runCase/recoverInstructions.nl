---
description: recoverInstructions domain stage for Appointment Preparation Service.
args:
  goal: string
  records: AppointmentPreparation[]
  evidence: EvidenceItem[]
returns: StageResultPreparationRequirement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPreparationRequirement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Parse the approved instruction set into actions, timing, exceptions, and contact routes. Preserve clinical wording where transformation could change meaning, and identify ambiguity for staff resolution rather than interpreting medical instructions independently. The stage gate is: Every consequential instruction and exception remains traceable to the approved source.
