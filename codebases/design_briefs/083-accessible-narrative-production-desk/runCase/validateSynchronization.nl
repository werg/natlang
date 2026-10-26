---
description: validateSynchronization domain stage for Accessible Narrative Production Desk.
args:
  goal: string
  records: AccessibleSegment[]
  evidence: EvidenceItem[]
returns: StageResultAccessibilityCheck
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAccessibilityCheck values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Run exact timing, overlap, reading-rate, and format checks, then review narrative completeness and usability. Inspect transitions and dense scenes where local compliance may still produce an inaccessible overall experience. The stage gate is: Exact synchronization passes and essential information remains understandable across channels.
