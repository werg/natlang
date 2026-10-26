---
description: mapPositions domain stage for Community Agreement Mediator.
args:
  goal: string
  records: ExistingAgreement[]
  evidence: EvidenceItem[]
returns: StageResultMediationPosition
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMediationPosition values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Summarize each participant’s position in terms they can review and correct. Separate factual disputes, preferences, and nonnegotiable limits, and do not infer hidden motives or psychological explanations from their communication style. The stage gate is: Each position is confirmed or clearly marked as an unconfirmed interpretation.
