---
description: assessOverlap domain stage for Participatory Budgeting Facilitator.
args:
  goal: string
  records: CivicProposal[]
  evidence: EvidenceItem[]
returns: StageResultProposalRelationship
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftProposalRelationship values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare proposals by intended benefit, location, and required work. Identify genuine duplicates and complementary bundles without merging projects whose similar wording masks different communities or uses; obtain proposer review for material combinations. The stage gate is: Merging preserves distinct intents and does not erase minority proposals.
