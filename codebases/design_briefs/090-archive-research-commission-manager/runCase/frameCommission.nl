---
description: frameCommission domain stage for Archive Research Commission Manager.
args:
  goal: string
  records: ArchiveCommission[]
  evidence: EvidenceItem[]
returns: StageResultResearchCommissionScope
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftResearchCommissionScope values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Interpret the requester’s goal and decompose it into answerable historical questions. Identify what available collections could establish, distinguish factual lookup from interpretive synthesis, and agree a bounded effort and stopping rule before searching indefinitely. The stage gate is: Each subquestion has a plausible evidence path and an explicit inferential limit.
