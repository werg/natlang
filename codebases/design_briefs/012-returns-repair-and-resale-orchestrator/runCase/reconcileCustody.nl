---
args: { goal: GoalContract, policy: PolicySelection, candidates: RepairPlan[], evidence: EvidencePage[] }
returns: StageResultCustodyEvent
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Track collection, receipt, repair transfer, and return as distinct custody events. Compare condition at each handoff and isolate new discrepancies rather than attributing all damage to the customer or overwriting the original inspection.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CustodyEvent fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Inventory changes require a confirmed custody event; serial-number mismatches block transfer.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
