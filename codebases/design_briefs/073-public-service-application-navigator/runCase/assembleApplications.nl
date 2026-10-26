---
description: assembleApplications domain stage for Public Service Application Navigator.
args:
  goal: string
  records: NavigationQuestion[]
  evidence: EvidenceItem[]
returns: StageResultApplicationTask
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftApplicationTask values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Convert chosen pathways into document, form, appointment, and submission tasks. Check consistency across forms, preserve the applicant’s own statements, and request review for uncertain answers rather than fabricating details to make a form complete. The stage gate is: Application content is evidence-backed and explicitly confirmed by the applicant where required.
