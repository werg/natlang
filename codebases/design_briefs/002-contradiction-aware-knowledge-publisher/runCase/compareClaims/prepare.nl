---
args: { candidate: ScopedClaim, goal: GoalContract, policy: PolicySelection, evidence: EvidencePage[] }
returns: PreparedClaimConflict
---
1. Select evidence spans for this candidate's exact subject, applicable date, authority, jurisdiction or project scope, and relevant record version. Ground the interpretation in this domain episode: Two onboarding pages disagree about trial eligibility because one describes a retired plan and one contains an actual typo.
2. Include supporting and conflicting passages. State searched partitions and whether each is complete; retrieval absence is not nonexistence outside that universe.
3. Identify plausible interpretations and list the exact fields the source does not establish.
4. Return the unchanged candidate identity, bounded evidence, alternatives, and missing fields.
