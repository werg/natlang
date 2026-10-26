---
description: interpretIntent domain stage for Persistent World Game Director.
args:
  goal: string
  records: GameWorld[]
  evidence: EvidenceItem[]
returns: StageResultPlayerIntent
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPlayerIntent values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Interpret player language in the visible world context and map it to permitted action types. Distinguish speech, intention, and executed action, ask only consequential clarifications, and avoid granting capabilities because a player narrates success confidently. The stage gate is: Intent resolves to a legal action proposal or remains unexecuted.
