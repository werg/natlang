---
description: selectEdits domain stage for Documentary Assembly Studio.
args:
  goal: string
  records: DocumentaryStructure[]
  evidence: EvidenceItem[]
returns: StageResultDocumentaryEdit
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftDocumentaryEdit values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assemble candidate sequences and transitions under exact duration constraints. Inspect how cuts, adjacency, and narration change the apparent meaning of statements, preserving necessary context and identifying misleading implications even when every individual clip is authentic. The stage gate is: No materially misleading quote edit or unsupported causal implication.
