---
description: reviewContent domain stage for Health Record Request Fulfillment Workbench.
args:
  goal: string
  records: RecordSelection[]
  evidence: EvidenceItem[]
returns: StageResultRecordDisclosureFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecordDisclosureFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assess responsiveness and identify third-party or otherwise restricted material under supplied disclosure rules. Preserve clinically relevant context where authorized, and route nuanced disclosure judgments to qualified records reviewers rather than making autonomous legal decisions. The stage gate is: Every exclusion or redaction has a supplied policy basis and review status.
