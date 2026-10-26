---
args: { goal: GoalContract, policy: PolicySelection, candidates: NegativeFinding[], evidence: EvidencePage[] }
returns: StageResultFailureExplanation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Construct competing explanations grounded in observed deviations and results. Separate demonstrated causes from plausible hypotheses, seek contradictory observations, and identify shared mechanisms across attempts without assuming all similar outcomes have one cause.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual FailureExplanation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each causal claim has evidence or an explicit hypothetical status.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
