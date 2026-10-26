---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReviewPlan[], evidence: EvidencePage[] }
returns: StageResultResolutionDelta
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconcile owner responses against the pending question and current evidence version. Distinguish approval of wording from approval of the decision; append the adopted resolution and invalidate only consumers of the replaced version.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResolutionDelta fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Signed owner authority and expected version must match before commit.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
