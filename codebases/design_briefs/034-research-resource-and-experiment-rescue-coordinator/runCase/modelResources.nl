---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExperimentRequirement[], evidence: EvidencePage[] }
returns: StageResultResourceCapability
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconcile calendars, instrument status, maintenance records, and operator qualifications. Distinguish booked availability from actual readiness and identify resources that appear interchangeable but differ in capability or validated use.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResourceCapability fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact capacity and qualification checks precede scheduling.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
