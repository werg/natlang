---
args: { goal: GoalContract, policy: PolicySelection, candidates: SubstituteCandidate[], evidence: EvidencePage[] }
returns: StageResultCompatibilityFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare every essential requirement against documented candidate evidence and approved conversion rules. Check interacting properties and interface tolerances; preserve unknowns explicitly and reject candidates whose apparent equivalence depends on an unsupported assumption.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CompatibilityFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: All essential properties pass independently; one critical failure disqualifies the candidate.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
