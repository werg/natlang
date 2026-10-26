---
args: { goal: GoalContract, policy: PolicySelection, candidates: ContainmentPlan[], evidence: EvidencePage[] }
returns: StageResultCorrectiveAction
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Evaluate the supplier’s explanation against incident evidence and rival mechanisms. Separate immediate correction from prevention of recurrence, identify testable claims, and request evidence that the proposed change addresses the actual mechanism rather than just the symptoms.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CorrectiveAction fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: A plausible narrative alone cannot close the causal investigation.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
