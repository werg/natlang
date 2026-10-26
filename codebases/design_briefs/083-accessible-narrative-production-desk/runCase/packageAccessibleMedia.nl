---
description: packageAccessibleMedia domain stage for Accessible Narrative Production Desk.
args:
  goal: string
  records: AccessibilityCheck[]
  evidence: EvidenceItem[]
returns: StageResultAccessibilityPackage
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAccessibilityPackage values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assemble coordinated format-specific files with stable segment IDs and the correct source revision. Verify player or delivery compatibility, preserve reviewer notes, and update only affected segments when the underlying edit changes. The stage gate is: All deliverables align to the same approved media timeline and version.
