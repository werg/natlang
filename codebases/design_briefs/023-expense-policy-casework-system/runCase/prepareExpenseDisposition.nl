---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExpenseQuestion[], evidence: EvidencePage[] }
returns: StageResultExpenseDisposition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Produce line-level recommendations and a clear explanation of reductions or holds. Support independent review and appeals with new evidence, then reconcile the approved amount to the reimbursement service without changing the historical original claim.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExpenseDisposition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Segregated approval and exact total conservation across paid, rejected, and held portions.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
