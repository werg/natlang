---
description: extractPermissions domain stage for Collection Rights And Consent Steward.
args:
  goal: string
  records: CollectionUse[]
  evidence: EvidenceItem[]
returns: StageResultCollectionPermission
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCollectionPermission values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read donation agreements, licenses, consent statements, and amendments to identify permissions and restrictions. Distinguish ownership of a physical object from rights to reproduce or publish it, and preserve missing or conflicting documentation as uncertainty. The stage gate is: Absence of a restriction record is not treated as permission.
