---
args: { goal: GoalContract, policy: PolicySelection, candidates: ChangeImpact[], evidence: EvidencePage[] }
returns: StageResultReviewPlan
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Traverse only necessary dependencies first, then inspect supporting edges for materiality. Group reviews sharing an unresolved premise, formulate a decision-ready question, and avoid automatically cancelling downstream work that remains independently justified.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ReviewPlan fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No dependency cycle silently flattened; each affected owner receives one deduplicated task.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
