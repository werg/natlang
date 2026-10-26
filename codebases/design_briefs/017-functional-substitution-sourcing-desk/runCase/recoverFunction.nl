---
args: { goal: GoalContract, policy: PolicySelection, candidates: SubstitutionRequest[], evidence: EvidencePage[] }
returns: StageResultFunctionalRequirement
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret why the original item is used, including interfaces, operating environment, and failure consequences. Separate critical properties from historical purchasing habits, and ask the requester to confirm distinctions that determine compatibility or safety.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual FunctionalRequirement fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every essential property has a use-based rationale; unknown critical properties block qualification.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
