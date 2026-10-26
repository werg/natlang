---
description: trackArcs domain stage for Serial Fiction Continuity Room.
args:
  goal: string
  records: CanonFact[]
  evidence: EvidenceItem[]
returns: StageResultStoryObligation
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftStoryObligation values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Identify unresolved setups, character goals, relationship changes, and narrative promises across installments. Distinguish intentional mystery from accidental omission, and preserve motivations that constrain plausible future actions without treating characters as mechanically fixed. The stage gate is: Arc claims are grounded in the actual text or approved author intent.
