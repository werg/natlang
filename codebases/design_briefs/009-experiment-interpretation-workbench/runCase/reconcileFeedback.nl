---
args: { goal: GoalContract, policy: PolicySelection, candidates: ResultClaim[], evidence: EvidencePage[] }
returns: StageResultMechanismFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare qualitative accounts with result patterns to formulate plausible mechanisms, not causal proof. Seek dissatisfied successes and satisfied failures that challenge the simple explanation, and identify evidence capable of distinguishing rival accounts.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual MechanismFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Anecdotes cannot override valid numerical results or establish unmeasured causality.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
