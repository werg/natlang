---
description: reconcileFindings domain stage for Archive Research Commission Manager.
args:
  goal: string
  records: ArchiveLead[]
  evidence: EvidenceItem[]
returns: StageResultArchiveFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftArchiveFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare evidence across sources while preserving chronology, attribution, and disagreement. State what was found and what remains uncertain, and do not turn an unsuccessful search in limited collections into proof that an event never occurred. The stage gate is: Conclusions remain within actual evidence and search coverage.
