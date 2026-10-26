---
description: composeChecklist domain stage for Appointment Preparation Service.
args:
  goal: string
  records: PreparationContext[]
  evidence: EvidenceItem[]
returns: StageResultPreparationChecklist
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPreparationChecklist values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Produce an accessible checklist using approved terminology and exact timing services. Adapt layout, language level, or format while preserving all clinical conditions; separate logistical suggestions from clinician instructions and flag unresolved questions visibly. The stage gate is: Semantic fidelity to approved instructions passes independent review.
