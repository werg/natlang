---
description: constructArguments domain stage for Documentary Assembly Studio.
args:
  goal: string
  records: DocumentarySegment[]
  evidence: EvidenceItem[]
returns: StageResultDocumentaryStructure
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftDocumentaryStructure values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Develop alternative narrative structures around the editorial question and available evidence. Include relevant contradictions and uncertainty, distinguish thematic juxtaposition from causal implication, and avoid inventing connective facts to make the story more dramatic. The stage gate is: Narrative argument remains supportable by the selected material.
