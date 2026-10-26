---
description: modelDramaturgy domain stage for Immersive Theatre Stage Manager.
args:
  goal: string
  records: ImmersiveProduction[]
  evidence: EvidenceItem[]
returns: StageResultTheatreScene
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTheatreScene values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the script and production notes to identify scene goals, knowledge reveals, dependencies, and branching choices. Distinguish fixed dramatic obligations from improvisable material and preserve what each character and audience group can know at that point. The stage gate is: Scene transitions respect knowledge and narrative prerequisites.
