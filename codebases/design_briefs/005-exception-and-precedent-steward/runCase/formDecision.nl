---
args: { goal: GoalContract, policy: PolicySelection, candidates: ApplicabilityFinding[], evidence: EvidencePage[] }
returns: StageResultExceptionDisposition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct a scoped recommendation with the original rule, proposed deviation, continuing controls, and expiration. Explain consistency with comparable cases and explicitly surface any departure requiring a higher authority.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExceptionDisposition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Required reviewer and expiration present; surviving safeguards not accidentally waived.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
