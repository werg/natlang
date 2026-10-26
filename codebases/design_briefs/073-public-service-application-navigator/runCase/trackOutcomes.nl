---
description: trackOutcomes domain stage for Public Service Application Navigator.
args:
  goal: string
  records: ApplicationTask[]
  evidence: EvidenceItem[]
returns: StageResultServiceApplicationStatus
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftServiceApplicationStatus values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Record actual submission and agency responses, distinguish receipt from acceptance or eligibility, and formulate next actions for missing evidence or appeal information under supplied procedures. Preserve the agency’s authority and the applicant’s choices. The stage gate is: No claim of benefit approval without authoritative confirmation.
