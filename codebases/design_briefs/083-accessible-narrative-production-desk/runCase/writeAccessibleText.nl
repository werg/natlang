---
description: writeAccessibleText domain stage for Accessible Narrative Production Desk.
args:
  goal: string
  records: AccessibilityChannelPlan[]
  evidence: EvidenceItem[]
returns: StageResultAccessibleSegment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAccessibleSegment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Formulate concise, natural accessible text appropriate to the channel and supplied style guidance. Describe observable actions rather than inferred motives, identify speakers or meaningful sounds where needed, and preserve narrative tone without adding unsupported interpretation. The stage gate is: Text is accurate, appropriately timed, and does not spoil future information.
