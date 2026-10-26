---
description: interpretSituation domain stage for Public Service Application Navigator.
args:
  goal: string
  records: ServiceNavigation[]
  evidence: EvidenceItem[]
returns: StageResultServiceSituation
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftServiceSituation values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the person’s stated situation and supplied documents to identify needs and established facts. Avoid inferring sensitive characteristics, distinguish missing information from ineligibility, and route urgent concerns according to approved service procedures. The stage gate is: Only volunteered or authorized evidence informs the situation model.
