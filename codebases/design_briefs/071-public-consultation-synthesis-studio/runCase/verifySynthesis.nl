---
description: verifySynthesis domain stage for Public Consultation Synthesis Studio.
args:
  goal: string
  records: ConsultationResponseOption[]
  evidence: EvidenceItem[]
returns: StageResultConsultationReport
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftConsultationReport values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Audit the synthesis against sampled submissions and all identified minority themes, reconcile exact counts, and prepare an accessible report. Preserve methodological limitations and provide traceable examples without exposing unnecessary personal information. The stage gate is: No material theme omission, misleading denominator, or privacy leak in published evidence.
