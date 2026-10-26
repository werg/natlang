---
description: trackAcknowledgment domain stage for Appointment Preparation Service.
args:
  goal: string
  records: PreparationQuestion[]
  evidence: EvidenceItem[]
returns: StageResultPreparationStatus
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPreparationStatus values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Record receipt, acknowledgment, and reported completion distinctly, respecting contact preferences and quiet periods. Escalate unresolved administrative barriers under supplied procedures without assuming acknowledgment proves comprehension or medical readiness. The stage gate is: No false readiness claim and no unapproved clinical recommendation.
