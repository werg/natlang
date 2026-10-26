---
args: { goal: GoalContract, policy: PolicySelection, candidates: OnboardingTask[], evidence: EvidencePage[] }
returns: StageResultProgressFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read system events and participant updates against each task’s completion contract. Distinguish an attempted import from a reconciled migration and attendance from demonstrated setup capability; request the smallest missing verification artifact.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ProgressFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Completion requires evidence, not a percentage reported in free text.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
