---
args: { goal: GoalContract, policy: PolicySelection, candidates: Commitment[], evidence: EvidencePage[] }
returns: StageResultWorkStatus
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Match outputs and acknowledgments to each promised outcome. A sent file is not accepted delivery when acceptance was required; cancellation must come from an authorized party, and incomplete evidence produces a status question.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual WorkStatus fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Completion requires the specified evidence, not merely a past due date or optimistic message.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
