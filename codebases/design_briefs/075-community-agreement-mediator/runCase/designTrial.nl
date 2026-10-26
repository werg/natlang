---
description: designTrial domain stage for Community Agreement Mediator.
args:
  goal: string
  records: CommunityOption[]
  evidence: EvidenceItem[]
returns: StageResultAgreementTrial
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAgreementTrial values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Turn a promising voluntary option into a reversible trial with clear responsibilities, resource rules, and review criteria. Specify how participants may report problems or leave the trial, and avoid making temporary arrangements permanent by default. The stage gate is: Trial requires explicit consent and clear reversal or exit conditions.
