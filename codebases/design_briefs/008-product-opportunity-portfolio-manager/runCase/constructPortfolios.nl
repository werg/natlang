---
args: { goal: GoalContract, policy: PolicySelection, candidates: OpportunityEstimate[], evidence: EvidencePage[] }
returns: StageResultPortfolioOption
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Submit eligible options and exact constraints to a portfolio solver, then interpret feasible alternatives against the supplied strategy. Explain who benefits and what is deferred; generate contingency portfolios for major assumption changes rather than a single opaque ranking.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PortfolioOption fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact budget and capacity feasibility; ranking rationale uses supplied priorities only.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
