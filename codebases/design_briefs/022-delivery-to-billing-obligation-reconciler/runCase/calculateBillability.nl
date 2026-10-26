---
args: { goal: GoalContract, policy: PolicySelection, candidates: AcceptanceFinding[], evidence: EvidencePage[] }
returns: StageResultBillableAllocation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Submit established milestone states and exact schedules to the billing calculator. Account for prior invoices, credits, caps, and partial acceptance; explain remaining holds without inventing a commercial compromise not authorized by the contract.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual BillableAllocation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No overbilling, duplicate milestone consumption, or amount based on an unresolved trigger.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
