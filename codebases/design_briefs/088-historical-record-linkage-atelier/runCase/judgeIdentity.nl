---
description: judgeIdentity domain stage for Historical Record Linkage Atelier.
args:
  goal: string
  records: HistoricalCandidate[]
  evidence: EvidenceItem[]
returns: StageResultHistoricalIdentityFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftHistoricalIdentityFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare evidence for identity versus kinship, institutional succession, or coincidence. Weigh source reliability only within its actual scope, preserve competing candidates, and avoid treating a plausible story as proof of identity. The stage gate is: Established links require discriminating evidence and compatible chronology.
