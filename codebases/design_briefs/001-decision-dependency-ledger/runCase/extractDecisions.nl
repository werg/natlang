---
args: { goal: GoalContract, policy: PolicySelection, candidates: DecisionCase[], evidence: EvidencePage[] }
returns: StageResultDecisionAssertion
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Segment meetings at topic boundaries; distinguish a suggestion, assent, and an authorized resolution. Link explicit objections and preserve the speaker rather than attributing a collective decision from an unchallenged statement.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DecisionAssertion fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No accepted resolution without attributable acceptance and scope evidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
