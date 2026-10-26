---
description: commissionPieces domain stage for Magazine Commissioning And Issue Planning Desk.
args:
  goal: string
  records: IssueCoverage[]
  evidence: EvidenceItem[]
returns: StageResultCommissionBrief
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCommissionBrief values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Draft precise commissions specifying angle, scope, length, reporting obligations, rights, and deadlines. Identify dependencies such as interviews or photography, obtain contributor acceptance, and avoid assuming a pitch implies agreement to changed terms. The stage gate is: Contributor accepts the exact scope and commercial conditions before work is committed.
