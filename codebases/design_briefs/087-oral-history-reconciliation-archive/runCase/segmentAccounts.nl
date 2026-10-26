---
description: segmentAccounts domain stage for Oral History Reconciliation Archive.
args:
  goal: string
  records: OralHistoryProject[]
  evidence: EvidenceItem[]
returns: StageResultOralHistoryAccount
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftOralHistoryAccount values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Segment interviews into attributable accounts while preserving surrounding context and the speaker’s expressions of uncertainty. Distinguish firsthand recollection, hearsay, later interpretation, and quoted speech rather than presenting them all as equivalent historical evidence. The stage gate is: Attribution and evidentiary character remain explicit.
