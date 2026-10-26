---
args: { goal: GoalContract, policy: PolicySelection, candidates: EnvironmentConstraint[], evidence: EvidencePage[] }
returns: StageResultOnboardingTask
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Decompose outcomes into configuration, data, training, and acceptance tasks. Order dependencies, allocate ownership explicitly, and create reversible pilots before broad changes; identify shared prerequisites that would otherwise generate duplicated requests.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OnboardingTask fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Task graph is acyclic and every outcome is covered by an executable path.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
