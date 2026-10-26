---
description: planChannels domain stage for Accessible Narrative Production Desk.
args:
  goal: string
  records: NarrativeInformation[]
  evidence: EvidenceItem[]
returns: StageResultAccessibilityChannelPlan
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAccessibilityChannelPlan values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Allocate information across captions, descriptions, transcripts, and supplementary explanations. Identify timing gaps and channel conflicts, prioritize essential meaning, and avoid duplicating dialogue in a way that obscures other necessary information. The stage gate is: Channel plan preserves essential information within actual timing and capacity constraints.
