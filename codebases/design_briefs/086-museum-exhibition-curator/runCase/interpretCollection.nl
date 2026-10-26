---
description: interpretCollection domain stage for Museum Exhibition Curator.
args:
  goal: string
  records: ExhibitionProject[]
  evidence: EvidenceItem[]
returns: StageResultCuratorialEvidence
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCuratorialEvidence values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read artifact records, scholarship, provenance, and conservation guidance to identify what can be responsibly said and displayed. Separate documented facts from contested interpretations and preserve restrictions on handling, light, duration, or cultural presentation. The stage gate is: Every proposed fact and display use has evidence and applicable permission.
