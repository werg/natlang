---
args: { goal: GoalContract, policy: PolicySelection, candidates: Requirement[], evidence: EvidencePage[] }
returns: StageResultAcceptanceScenario
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Generate ordinary, boundary, and adverse scenarios, then replay the specification through the available simulator. Use counterexamples to reopen only the implicated agreements, preserving unrelated acceptance and recording revised stakeholder acknowledgment.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AcceptanceScenario fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every mandatory requirement has a discriminating scenario and no blocking scenario remains unexplained.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
