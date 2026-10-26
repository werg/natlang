---
args: { candidate: ActivityEvidence, goal: GoalContract, policy: PolicySelection, evidence: EvidencePage[] }
returns: PreparedGrantNarrative
---
1. Select evidence spans for this candidate's exact subject, applicable date, authority, jurisdiction or project scope, and relevant record version. Ground the interpretation in this domain episode: A project meets participation targets but lacks evidence for claimed long-term impact, and a shared contractor cost appears in two grants.
2. Include supporting and conflicting passages. State searched partitions and whether each is complete; retrieval absence is not nonexistence outside that universe.
3. Identify plausible interpretations and list the exact fields the source does not establish.
4. Return the unchanged candidate identity, bounded evidence, alternatives, and missing fields.
