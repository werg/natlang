---
description: modelReaderTasks domain stage for Plain Language Transformation Studio.
args:
  goal: string
  records: MeaningUnit[]
  evidence: EvidenceItem[]
returns: StageResultReaderTask
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftReaderTask values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Interpret what the intended reader must understand or do using supplied audience guidance. Identify decision points and necessary background without stereotyping ability, and distinguish information that can be reorganized from information that cannot be omitted. The stage gate is: Reader tasks cover all consequential uses of the document.
