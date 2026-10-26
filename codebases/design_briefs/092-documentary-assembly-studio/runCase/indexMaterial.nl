---
description: indexMaterial domain stage for Documentary Assembly Studio.
args:
  goal: string
  records: DocumentaryProject[]
  evidence: EvidenceItem[]
returns: StageResultDocumentarySegment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftDocumentarySegment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Segment interviews and archive descriptions while preserving timecodes, speaker context, and rights. Separate a speaker’s claim from independently established fact and retain qualifiers or questions that affect the meaning of an extract. The stage gate is: Every segment has source timecodes, context, and permitted-use status.
