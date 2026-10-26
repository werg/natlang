---
description: recoverAgreements domain stage for Community Agreement Mediator.
args:
  goal: string
  records: CommunityDispute[]
  evidence: EvidenceItem[]
returns: StageResultExistingAgreement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExistingAgreement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconstruct prior agreements from records and participant accounts, distinguishing signed or acknowledged terms from expectations and informal habits. Preserve disagreement about what was agreed and avoid treating one participant’s confident recollection as authoritative. The stage gate is: Agreement status and evidence are explicit.
