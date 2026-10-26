---
description: validateLocale domain stage for Localization Release Manager.
args:
  goal: string
  records: LocalizedSegment[]
  evidence: EvidenceItem[]
returns: StageResultLocaleQualityFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftLocaleQualityFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Run exact placeholder, markup, length, plural-form, and link checks, then review semantic fidelity and naturalness independently. Inspect cross-segment consistency and context screenshots or metadata, distinguishing harmless variation from contradictory terminology or behavior. The stage gate is: Exact structural gates pass and no unresolved material semantic defect remains.
