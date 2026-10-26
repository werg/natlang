---
description: formNarratives domain stage for Museum Exhibition Curator.
args:
  goal: string
  records: CuratorialEvidence[]
  evidence: EvidenceItem[]
returns: StageResultExhibitionArgument
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExhibitionArgument values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate distinct interpretive arguments that connect artifacts meaningfully rather than merely grouping them by topic. Include alternative perspectives and uncertainties, identify missing evidence, and avoid inventing historical continuity to make a more dramatic story. The stage gate is: Narrative claims remain supported and contested interpretations are represented fairly.
