---
description: interpretCircumstances domain stage for Appointment Preparation Service.
args:
  goal: string
  records: PreparationRequirement[]
  evidence: EvidenceItem[]
returns: StageResultPreparationContext
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPreparationContext values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the patient’s stated practical circumstances and communication preferences without inferring diagnoses or capability. Identify logistical barriers and questions requiring staff attention, and verify authority before involving a representative. The stage gate is: Context uses volunteered facts and does not change the clinical plan.
