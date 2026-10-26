---
description: chooseWorldActions domain stage for Persistent World Game Director.
args:
  goal: string
  records: CharacterView[]
  evidence: EvidenceItem[]
returns: StageResultCharacterAction
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCharacterAction values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Choose among exact legal actions in a way consistent with the character’s knowledge, goals, and relationships. Generate dialogue and creative alternatives within world and content boundaries, and avoid overriding resource or location constraints for narrative convenience. The stage gate is: Action is legal and dialogue does not reveal unknown or restricted facts.
