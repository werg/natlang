---
description: assessCompleteness domain stage for Referral Packet Completeness Manager.
args:
  goal: string
  records: ReferralRecordMatch[]
  evidence: EvidenceItem[]
returns: StageResultReferralGap
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftReferralGap values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare each requirement with submitted documents, checking scope, dates, signatures, and readability. Distinguish document presence from actual required content and route clinical-content questions to authorized staff rather than interpreting them autonomously. The stage gate is: Every administrative requirement is met or explicitly unresolved.
