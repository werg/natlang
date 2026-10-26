---
description: linkCampaigns domain stage for Public Consultation Synthesis Studio.
args:
  goal: string
  records: ConsultationArgument[]
  evidence: EvidenceItem[]
returns: StageResultSubmissionRelation
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftSubmissionRelation values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Detect copied or campaign-template language while preserving independently added reasons and legitimate participation counts. Distinguish duplicate ingestion from multiple people submitting the same position, and report volume and argument diversity separately. The stage gate is: Template use does not invalidate a participant’s submission or erase unique content.
