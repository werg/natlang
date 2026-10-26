---
description: reviewOutcome domain stage for Community Agreement Mediator.
args:
  goal: string
  records: AgreementTrial[]
  evidence: EvidenceItem[]
returns: StageResultAgreementReview
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAgreementReview values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare actual use and participant feedback with the trial criteria, preserving divergent experiences. Propose continuation, revision, or termination, and record only the terms participants explicitly accept rather than treating silence as agreement. The stage gate is: Final agreement reflects actual consent and does not erase unresolved disagreement.
