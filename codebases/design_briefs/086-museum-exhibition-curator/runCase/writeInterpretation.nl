---
description: writeInterpretation domain stage for Museum Exhibition Curator.
args:
  goal: string
  records: ExhibitionZone[]
  evidence: EvidenceItem[]
returns: StageResultExhibitionLabel
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExhibitionLabel values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Write labels, introductions, and supporting text that are engaging without overstating evidence. Distinguish fact, attribution, and interpretation, preserve culturally sensitive context, and keep terminology consistent across the visitor journey. The stage gate is: Every material claim is supported or explicitly framed as interpretation.
