---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExpenseFinding[], evidence: EvidencePage[] }
returns: StageResultExpenseQuestion
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Group unresolved questions to minimize claimant burden while keeping each decision testable. Ask only for evidence necessary to determine eligibility, explain the applicable rule, and route privacy-sensitive material through restricted upload channels.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExpenseQuestion fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Do not request information already available or unrelated personal records.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
