---
description: modelExperience domain stage for Escape Room Foundry.
args:
  goal: string
  records: EscapeRoomProject[]
  evidence: EvidenceItem[]
returns: StageResultEscapeConstraint
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEscapeConstraint values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Interpret the creative brief, venue, prop capabilities, audience needs, and approved safety rules. Separate dramatic fiction from actual participant constraints, preserve unrestricted emergency exit and staff intervention, and identify unsupported physical mechanisms before puzzle design. The stage gate is: No real confinement, unsafe mechanism, or inaccessible mandatory path under supplied requirements.
