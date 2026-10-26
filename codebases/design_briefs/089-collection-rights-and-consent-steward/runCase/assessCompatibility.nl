---
description: assessCompatibility domain stage for Collection Rights And Consent Steward.
args:
  goal: string
  records: ProposedCollectionUse[]
  evidence: EvidenceItem[]
returns: StageResultRightsFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRightsFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare each proposed act with supplied rights and consent conditions, including interactions among multiple rightsholders. Preserve scoped exceptions and refer ambiguous legal interpretation to the rights officer rather than selecting the most permissive reading. The stage gate is: Every allowed use has positive authority and all applicable conditions are satisfied.
