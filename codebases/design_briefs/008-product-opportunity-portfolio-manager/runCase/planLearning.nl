---
args: { goal: GoalContract, policy: PolicySelection, candidates: PortfolioOption[], evidence: EvidencePage[] }
returns: StageResultOpportunityExperiment
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Choose the cheapest credible experiment capable of changing a portfolio decision. State how positive, negative, and inconclusive results would alter the selected work, and schedule learning before irreversible dependent commitments wherever feasible.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OpportunityExperiment fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: An experiment has an actionable decision rule and does not merely collect agreeable feedback.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
