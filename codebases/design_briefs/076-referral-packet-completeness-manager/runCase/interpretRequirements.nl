---
description: interpretRequirements domain stage for Referral Packet Completeness Manager.
args:
  goal: string
  records: ReferralCase[]
  evidence: EvidenceItem[]
returns: StageResultReferralRequirement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftReferralRequirement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the receiving service’s supplied administrative requirements and the clinician’s referral instructions. Identify required records, forms, dates, and signatures while preserving clinician-provided priority and avoiding any independent judgment about clinical necessity. The stage gate is: Requirements are current, service-specific, and administrative in scope.
