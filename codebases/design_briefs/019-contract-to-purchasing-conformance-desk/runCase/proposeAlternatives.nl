---
args: { goal: GoalContract, policy: PolicySelection, candidates: ConformanceFinding[], evidence: EvidencePage[] }
returns: StageResultPurchasingOption
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct alternatives that satisfy the underlying need within the applicable agreement. Explain changed capabilities and total costs, and offer an exception route where compliant options would not meet essential requirements rather than silently weakening the request.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PurchasingOption fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Alternatives preserve confirmed essential constraints and exact commercial limits.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
