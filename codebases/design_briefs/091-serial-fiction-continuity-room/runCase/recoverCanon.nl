---
description: recoverCanon domain stage for Serial Fiction Continuity Room.
args:
  goal: string
  records: StorySeries[]
  evidence: EvidenceItem[]
returns: StageResultCanonFact
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCanonFact values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Extract events, relationships, world rules, and character knowledge from approved manuscripts and notes. Distinguish established canon from plans, unreliable narration, and character beliefs; preserve conflicting drafts without silently promoting one to truth. The stage gate is: Every canon fact has source and status, including belief versus world fact.
