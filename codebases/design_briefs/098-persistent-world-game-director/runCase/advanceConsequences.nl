---
description: advanceConsequences domain stage for Persistent World Game Director.
args:
  goal: string
  records: CharacterAction[]
  evidence: EvidenceItem[]
returns: StageResultWorldConsequence
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftWorldConsequence values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Apply actions through the exact world simulator, then interpret their social and narrative consequences without rewriting physical outcomes. Preserve causal history, update beliefs only through information channels, and record unresolved story obligations for later sessions. The stage gate is: Resource conservation, chronology, and knowledge propagation remain valid.
