---
description: manageEditorialWork domain stage for Magazine Commissioning And Issue Planning Desk.
args:
  goal: string
  records: CommissionBrief[]
  evidence: EvidenceItem[]
returns: StageResultEditorialTask
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEditorialTask values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare drafts with commissions and source evidence, formulate specific revisions, and coordinate editing, fact-checking, art, and legal or rights review where needed. Preserve author voice while distinguishing stylistic preference from factual or structural defects. The stage gate is: Required reporting and evidence obligations are fulfilled, not merely polished prose.
