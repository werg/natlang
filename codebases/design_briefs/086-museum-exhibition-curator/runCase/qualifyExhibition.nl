---
description: qualifyExhibition domain stage for Museum Exhibition Curator.
args:
  goal: string
  records: ExhibitionLabel[]
  evidence: EvidenceItem[]
returns: StageResultExhibitionRelease
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExhibitionRelease values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Review the complete exhibition for narrative coherence, factual support, rights, conservation, accessibility, and operational feasibility. Compare alternative arrangements, resolve conflicts with qualified owners, and freeze the approved content and installation plan together. The stage gate is: All required specialist approvals and exact physical constraints pass before installation.
