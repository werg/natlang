---
description: publishArchive domain stage for Oral History Reconciliation Archive.
args:
  goal: string
  records: OralHistoryQuestion[]
  evidence: EvidenceItem[]
returns: StageResultOralHistoryEdition
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftOralHistoryEdition values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assemble an attributed narrative and search index that preserve disagreement and access restrictions. Apply consent to quotations, identities, and derived summaries, and maintain edition history when participants correct or withdraw material. The stage gate is: No restricted content leaks through summaries, search snippets, or merged narratives.
