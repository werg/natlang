---
description: identifyPathways domain stage for Public Service Application Navigator.
args:
  goal: string
  records: ServiceSituation[]
  evidence: EvidenceItem[]
returns: StageResultServicePathway
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftServicePathway values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare known facts with supplied service descriptions and conditions to identify potentially relevant routes. Preserve multiple plausible pathways, explain uncertainty, and avoid presenting a preliminary match as an official eligibility decision. The stage gate is: Every pathway follows current supplied rules and clearly states unresolved conditions.
