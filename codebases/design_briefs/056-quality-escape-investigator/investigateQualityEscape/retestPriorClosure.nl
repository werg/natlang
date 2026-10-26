---
args: { prior: QualityClosure | null, actions: CorrectiveActionRecord[], defects: DefectObservation[] }
returns: EffectivenessReview
---
Compare recurrence evidence against the closure's claimed target, action versions, and verification refs. If recurrence is within the same causal scope, reopen the claim and identify failed or unverified actions. A closed record without a comparable test cannot establish effectiveness.
