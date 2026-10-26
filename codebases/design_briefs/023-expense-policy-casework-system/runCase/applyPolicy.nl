---
args: { goal: GoalContract, policy: PolicySelection, candidates: PurposeAllocation[], evidence: EvidencePage[] }
returns: StageResultExpenseFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Apply the policy edition effective for the expense and claimant context. Preserve category-specific limits, meal inclusions, and approval conditions; distinguish a documented exception from a persuasive but unauthorized justification.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExpenseFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact category and period limits; no retroactive policy substitution without authority.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
