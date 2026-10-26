---
description: assembleDossier domain stage for Archive Research Commission Manager.
args:
  goal: string
  records: ArchiveFinding[]
  evidence: EvidenceItem[]
returns: StageResultResearchDossier
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftResearchDossier values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Produce an evidence dossier with findings, source references, search history, unresolved leads, and reuse restrictions. Tailor the narrative to the commission while preserving methodological limits and separating the researcher’s interpretation from quoted or recorded facts. The stage gate is: Another researcher can reproduce the searched scope and distinguish evidence from inference.
