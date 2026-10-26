---
description: matchPatientRecords domain stage for Referral Packet Completeness Manager.
args:
  goal: string
  records: ReferralRequirement[]
  evidence: EvidenceItem[]
returns: StageResultReferralRecordMatch
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftReferralRecordMatch values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Verify patient and episode identity using approved exact matching and reviewed evidence. Distinguish similar names, historical identifiers, and unrelated episodes; quarantine mismatches before content is included in a packet. The stage gate is: No cross-patient attachment or unsupported identity merge.
