---
args: { goal: GoalContract, policy: PolicySelection, candidates: CapabilityMatch[], evidence: EvidencePage[] }
returns: StageResultContributionTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Group missing technical, financial, and implementation contributions by responsible team. Ask precise questions tied to requirements, specify acceptable evidence, and expose dependencies that could make the submission infeasible before the deadline.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ContributionTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every mandatory gap has an accountable owner or an explicit bid/no-bid escalation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
