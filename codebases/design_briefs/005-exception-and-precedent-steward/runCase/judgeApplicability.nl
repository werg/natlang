---
args: { goal: GoalContract, policy: PolicySelection, candidates: PrecedentCandidate[], evidence: EvidencePage[] }
returns: StageResultApplicabilityFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Test every limiting condition against current evidence and identify which differences change the rationale. If a decisive fact is unknown, request that fact; do not grant the exception by treating missing evidence as a match.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ApplicabilityFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: All mandatory conditions established and current authority confirmed before recommendation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
