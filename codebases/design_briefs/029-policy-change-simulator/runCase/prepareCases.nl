---
args: { goal: GoalContract, policy: PolicySelection, candidates: PolicyDelta[], evidence: EvidencePage[] }
returns: StageResultPolicyCaseFacts
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Normalize historical or synthetic cases into evidence-backed facts without adding missing information. Preserve uncertainty, policy-relevant context, and data-access restrictions; do not infer sensitive attributes merely to complete a simulation.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PolicyCaseFacts fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Case facts remain identical across baseline and candidate runs.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
