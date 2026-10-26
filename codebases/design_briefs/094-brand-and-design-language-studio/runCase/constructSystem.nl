---
description: constructSystem domain stage for Brand And Design Language Studio.
args:
  goal: string
  records: CreativeDirection[]
  evidence: EvidenceItem[]
returns: StageResultBrandSystem
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftBrandSystem values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Turn the selected direction into reusable principles and examples across likely contexts. Define how the system behaves in error messages, sensitive situations, and ordinary promotion, preserving consistency without requiring every asset to sound identical. The stage gate is: Rules are operationally usable and examples do not contradict them.
