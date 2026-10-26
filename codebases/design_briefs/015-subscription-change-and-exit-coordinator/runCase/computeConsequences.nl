---
args: { goal: GoalContract, policy: PolicySelection, candidates: ChangeIntent[], evidence: EvidencePage[] }
returns: StageResultChangeConsequence
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Consult authoritative entitlement and billing services for permitted dates and amounts, then explain consequences in plain language. Separate immediate access effects from future billing and data deletion; keep uncertain external balances explicitly unresolved.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ChangeConsequence fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Monetary values come from exact services; all material consequences are disclosed.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
