---
description: Reconcile participant, accepted submission, late inclusion and campaign-cluster counts as distinct measures.
args: { docket: ConsultationDocket; delta: SubmissionDelta; campaigns: CampaignRelationGraph }
returns: string[]
---
Count unique participant identities, accepted submissions, late submissions admitted under an explicit exception, duplicate ingestions, and template clusters separately. Preserve the denominator rule and count artifact reference. Never collapse multiple people because their text matches. Keep an unknown identity unresolved instead of estimating a unique-participant count.
