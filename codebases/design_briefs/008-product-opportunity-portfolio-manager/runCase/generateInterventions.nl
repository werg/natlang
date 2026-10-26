---
args: { goal: GoalContract, policy: PolicySelection, candidates: CustomerNeed[], evidence: EvidencePage[] }
returns: StageResultIntervention
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Propose materially different ways to improve each outcome, including process, service, and product changes. Explain the mechanism, affected users, and expected tradeoffs; avoid generating cosmetic variants that all depend on the same untested assumption.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual Intervention fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Candidate set includes a credible low-change baseline and distinct mechanisms.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
