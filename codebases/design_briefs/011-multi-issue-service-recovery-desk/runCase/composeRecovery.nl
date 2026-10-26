---
args: { goal: GoalContract, policy: PolicySelection, candidates: EntitlementFinding[], evidence: EvidencePage[] }
returns: StageResultRemedyBundle
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct alternative bundles that jointly solve the issues, accounting for dependencies and customer preferences. Detect double compensation for the same loss, but do not erase separate entitlements merely because they share one order.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RemedyBundle fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact amount limits and mutually exclusive remedies checked before presentation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
