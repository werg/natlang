---
description: modelNarrativeInformation domain stage for Accessible Narrative Production Desk.
args:
  goal: string
  records: AccessibleProduction[]
  evidence: EvidenceItem[]
returns: StageResultNarrativeInformation
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftNarrativeInformation values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the script and approved scene metadata to identify dialogue, sound, visual information, and narrative reveals. Distinguish essential understanding from decorative detail and preserve what the audience is meant to know at each moment. The stage gate is: No information is invented beyond supplied media evidence or revealed prematurely.
