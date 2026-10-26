---
args: { goal: GoalContract, policy: PolicySelection, candidates: ClauseNegotiation[], evidence: EvidencePage[] }
returns: StageResultNegotiationPosition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate stakeholder goals and the approved playbook into issue-level positions. Distinguish mandatory limits from preferences, identify dependencies between issues, and preserve disagreements that require a mandate decision before drafting a proposal.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual NegotiationPosition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every allowed concession has a supplied authority basis.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
