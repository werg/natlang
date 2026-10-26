---
description: interpretOffers domain stage for Negotiation Training Arena.
args:
  goal: string
  records: NegotiationRole[]
  evidence: EvidenceItem[]
returns: StageResultTrainingOffer
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTrainingOffer values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Parse natural-language offers into issue-level terms, conditions, and commitment status. Distinguish exploration from a firm offer and qualified acceptance from final agreement; ask clarifying questions when ambiguity would change the negotiated outcome. The stage gate is: Exact terms and authority limits are preserved before any simulated commitment.
