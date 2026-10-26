---
args: { candidate: DisclosureSelection, goal: GoalContract, policy: PolicySelection, evidence: EvidencePage[] }
returns: PreparedRedactionProposal
---
1. Select evidence spans for this candidate's exact subject, applicable date, authority, jurisdiction or project scope, and relevant record version. Ground the interpretation in this domain episode: Several documents separately reveal role, date, and location that jointly identify a protected third party despite name redaction.
2. Include supporting and conflicting passages. State searched partitions and whether each is complete; retrieval absence is not nonexistence outside that universe.
3. Identify plausible interpretations and list the exact fields the source does not establish.
4. Return the unchanged candidate identity, bounded evidence, alternatives, and missing fields.
