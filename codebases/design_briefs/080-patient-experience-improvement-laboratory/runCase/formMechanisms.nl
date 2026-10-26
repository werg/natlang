---
description: formMechanisms domain stage for Patient Experience Improvement Laboratory.
args:
  goal: string
  records: JourneyLink[]
  evidence: EvidenceItem[]
returns: StageResultServiceMechanism
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftServiceMechanism values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Identify plausible administrative mechanisms such as unclear instructions or failed callbacks. Seek contradictory cases and alternative explanations, distinguish correlation from causality, and define evidence needed to test the mechanism without making clinical claims. The stage gate is: Mechanisms remain appropriately qualified and source-supported.
