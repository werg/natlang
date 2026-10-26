---
description: generateFormats domain stage for Live Event Disruption Director.
args:
  goal: string
  records: EventDisruption[]
  evidence: EvidenceItem[]
returns: StageResultEventRecoveryOption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEventRecoveryOption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Invent alternative formats, sequences, or uses of available resources that preserve essential outcomes. Consider remote participation, moderated alternatives, or resequencing where approved, and disclose changes affecting access, consent, or contractual commitments. The stage gate is: Alternatives preserve hard safety and accessibility constraints.
