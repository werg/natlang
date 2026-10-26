---
description: normalizeRecords domain stage for Historical Record Linkage Atelier.
args:
  goal: string
  records: HistoricalLinkage[]
  evidence: EvidenceItem[]
returns: StageResultHistoricalObservation
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftHistoricalObservation values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Preserve original spellings, date expressions, and geographic descriptions while creating normalized search forms. Distinguish transcription uncertainty from historical variation, and avoid converting approximate dates or changing place names into falsely precise modern facts. The stage gate is: Original representation and uncertainty remain available.
