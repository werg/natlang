---
description: resolveReferences domain stage for Oral History Reconciliation Archive.
args:
  goal: string
  records: OralHistoryAccount[]
  evidence: EvidenceItem[]
returns: StageResultOralHistoryReference
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftOralHistoryReference values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Resolve names, places, and relationships using authorized contextual evidence, preserving aliases and uncertainty. Avoid merging people with similar names or assuming a pronoun’s referent when the interview context supports multiple interpretations. The stage gate is: Identity links are evidence-backed or remain provisional.
