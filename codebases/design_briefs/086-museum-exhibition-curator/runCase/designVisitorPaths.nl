---
description: designVisitorPaths domain stage for Museum Exhibition Curator.
args:
  goal: string
  records: ExhibitionArgument[]
  evidence: EvidenceItem[]
returns: StageResultExhibitionZone
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExhibitionZone values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Organize the exhibition into zones and visitor paths that support the argument and audience needs. Offer multiple levels of engagement, preserve accessibility and conservation constraints, and use exact space and flow services for feasibility checks. The stage gate is: Physical layout and artifact constraints are feasible under supplied venue data.
