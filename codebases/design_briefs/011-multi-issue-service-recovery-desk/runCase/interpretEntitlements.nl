---
args: { goal: GoalContract, policy: PolicySelection, candidates: ServiceIssue[], evidence: EvidencePage[] }
returns: StageResultEntitlementFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Match each issue to the applicable policy edition, purchased service, and documented exception. Separate mandatory entitlements from discretionary gestures; ask for missing decisive evidence without requiring information already present in the case.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual EntitlementFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No remedy outside delegated authority; original purchase dates determine applicable terms.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
