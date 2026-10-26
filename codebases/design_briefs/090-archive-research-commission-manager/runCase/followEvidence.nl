---
description: followEvidence domain stage for Archive Research Commission Manager.
args:
  goal: string
  records: ArchiveSearchPlan[]
  evidence: EvidenceItem[]
returns: StageResultArchiveLead
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftArchiveLead values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read retrieved records in context, extract relevant evidence, and formulate bounded follow-up leads. Distinguish direct evidence from references to missing sources, avoid circular citation chains, and stop low-value leads under the agreed effort budget. The stage gate is: Every lead has a question-specific rationale and source provenance.
