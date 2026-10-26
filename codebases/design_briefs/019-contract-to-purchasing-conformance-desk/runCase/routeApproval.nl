---
args: { goal: GoalContract, policy: PolicySelection, candidates: PurchasingOption[], evidence: EvidencePage[] }
returns: StageResultPurchaseDisposition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Build a review packet with scope evidence, pricing, alternatives, and unresolved conditions. Route to the correct authority based on cumulative commitment and exception type; recheck terms and request versions immediately before order release.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PurchaseDisposition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No split-order circumvention of thresholds; current scoped approval required.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
