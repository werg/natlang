---
description: formResponses domain stage for Public Consultation Synthesis Studio.
args:
  goal: string
  records: ConsultationTheme[]
  evidence: EvidenceItem[]
returns: StageResultConsultationResponseOption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftConsultationResponseOption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Draft evidence-linked response options for the accountable public body, including clarification, further investigation, or policy alternatives. Distinguish what the consultation establishes from questions requiring external evidence and avoid choosing political priorities not supplied by decision-makers. The stage gate is: Response accurately represents the concern and does not claim unsupported public consensus.
