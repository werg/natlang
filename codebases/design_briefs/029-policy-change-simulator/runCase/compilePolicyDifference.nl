---
args: { goal: GoalContract, policy: PolicySelection, candidates: PolicySimulation[], evidence: EvidencePage[] }
returns: StageResultPolicyDelta
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare baseline and proposed policies at the level of conditions, exceptions, authority, and consequences. Separate editorial changes from changed behavior and identify ambiguous provisions that prevent a determinate comparison.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PolicyDelta fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every behavioral delta has source support and preserved safeguards are explicit.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
