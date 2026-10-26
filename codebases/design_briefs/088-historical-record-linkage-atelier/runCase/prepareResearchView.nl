---
description: prepareResearchView domain stage for Historical Record Linkage Atelier.
args:
  goal: string
  records: HistoricalNetwork[]
  evidence: EvidenceItem[]
returns: StageResultHistoricalLinkageEdition
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftHistoricalLinkageEdition values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Publish a reviewable linkage view with source-level explanations and explicit alternatives. Support corrections and reversible splits, preserve stable source identifiers, and expose uncertainty to downstream researchers instead of exporting only a deceptively clean graph. The stage gate is: Accepted and provisional relationships remain distinguishable in every export.
