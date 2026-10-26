---
description: requestClearance domain stage for Collection Rights And Consent Steward.
args:
  goal: string
  records: RightsFinding[]
  evidence: EvidenceItem[]
returns: StageResultClearanceTask
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftClearanceTask values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Form a precise clearance request describing the actual use, audience, modifications, and duration. Track responses and conditions without treating informal enthusiasm as a license, and maintain restricted access to private donor or participant information. The stage gate is: Clearance closes a gap only with appropriate documented authority.
