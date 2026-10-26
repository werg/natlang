---
args: { goal: GoalContract, policy: PolicySelection, candidates: TrendSlice[], evidence: EvidencePage[] }
returns: StageResultInvestigationBrief
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Select unresolved explanations that would change a product decision. Formulate a bounded investigation with relevant cohorts, contradictory cases, and evidence needed to distinguish mechanisms rather than recommending the loudest requested feature.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual InvestigationBrief fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each task answers a named uncertainty and cites both support and known counterevidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
