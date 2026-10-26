---
args: { goal: GoalContract, policy: PolicySelection, candidates: ProgressFinding[], evidence: EvidencePage[] }
returns: StageResultOnboardingRevision
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: When dependencies fail or objectives change, construct feasible alternatives and disclose scope, timing, and effort consequences. Preserve accepted work, obtain approval for material changes, and replay end-to-end outcomes before declaring onboarding complete.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OnboardingRevision fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Revised plan preserves mandatory requirements or records explicit accepted scope reductions.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
