---
description: generateLinks domain stage for Historical Record Linkage Atelier.
args:
  goal: string
  records: HistoricalObservation[]
  evidence: EvidenceItem[]
returns: StageResultHistoricalCandidate
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftHistoricalCandidate values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate candidate links using names, chronology, places, relationships, and contextual descriptions. Include plausible aliases while preserving hard incompatibilities, and audit records excluded by blocking so unusual spellings do not disappear from consideration. The stage gate is: Candidate generation is bounded and recall is independently assessed.
