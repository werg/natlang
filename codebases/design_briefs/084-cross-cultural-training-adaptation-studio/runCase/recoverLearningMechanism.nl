---
description: recoverLearningMechanism domain stage for Cross Cultural Training Adaptation Studio.
args:
  goal: string
  records: TrainingAdaptation[]
  evidence: EvidenceItem[]
returns: StageResultTrainingMechanism
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTrainingMechanism values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Analyze the original training to identify the skill exercised by each scenario and the elements that merely provide context. Separate essential decision structure from names, settings, idioms, or assumptions that may be adapted without changing the learning objective. The stage gate is: Every scenario has an explicit instructional mechanism before adaptation.
