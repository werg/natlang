---
args: { sequence: ProposedRecoverySequence, impact: IncidentImpact, window: string }
returns: UnverifiedOutcomeCoverage
---
Match each affected cohort and function to an independent post-action observation inside the required duration. Return cohorts without a valid check, checks collected before stabilization, and user functions covered only by infrastructure metrics.
