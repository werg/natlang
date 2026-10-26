---
args: { goal: GoalContract, policy: PolicySelection, candidates: BillingCase[], evidence: EvidencePage[] }
returns: StageResultBillingCondition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read contract schedules and amendments to identify triggers, acceptance requirements, partial billing rules, and exclusions. Distinguish work performed from the specific event authorizing billing and retain conditions that apply only to particular deliverables.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual BillingCondition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every billable condition has a source and effective scope.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
