---
args: { goal: GoalContract, policy: PolicySelection, candidates: ConceptAlignment[], evidence: EvidencePage[] }
returns: StageResultRoleInterface
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare actual decision rights, escalation paths, and handoff expectations. Identify duplicated ownership and gaps between roles; propose responsibility agreements without assuming that similar job titles imply interchangeable authority.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual RoleInterface fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every required responsibility has either an accepted owner or an explicit unresolved gap.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
