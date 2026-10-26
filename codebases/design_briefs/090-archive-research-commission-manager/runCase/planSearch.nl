---
description: planSearch domain stage for Archive Research Commission Manager.
args:
  goal: string
  records: ResearchCommissionScope[]
  evidence: EvidenceItem[]
returns: StageResultArchiveSearchPlan
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftArchiveSearchPlan values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Select collections and search strategies using catalog structure, terminology variants, dates, and known access limits. Include negative search paths and record why a collection is relevant rather than relying only on keyword matches. The stage gate is: Search plan distinguishes accessible, inaccessible, unsearched, and exhausted collections.
