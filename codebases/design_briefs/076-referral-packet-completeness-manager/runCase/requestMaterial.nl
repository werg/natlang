---
description: requestMaterial domain stage for Referral Packet Completeness Manager.
args:
  goal: string
  records: ReferralGap[]
  evidence: EvidenceItem[]
returns: StageResultReferralRequest
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftReferralRequest values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Group missing-material requests by authorized source, ask only for necessary records, and preserve patient consent and permitted disclosure scope. Avoid repeated requests for existing evidence and track who is responsible for each unresolved gap. The stage gate is: Requests are minimal, authorized, and tied to a specific requirement.
