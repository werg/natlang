---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExpenseCase[], evidence: EvidencePage[] }
returns: StageResultExpenseLine
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Extract receipt lines and claimant explanations while preserving original currency and dates. Distinguish the merchant’s description from the claimed business purpose, and flag illegible or missing evidence rather than reconstructing amounts through guesswork.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExpenseLine fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact totals reconcile to source receipts; unverified values remain blocked.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
