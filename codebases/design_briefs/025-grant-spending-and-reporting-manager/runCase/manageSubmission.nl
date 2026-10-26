---
args: { goal: GoalContract, policy: PolicySelection, candidates: GrantNarrative[], evidence: EvidencePage[] }
returns: StageResultGrantSubmission
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Assemble the required forms, schedules, and evidence, obtain accountable approval, and record actual submission receipt. Track follow-up questions as obligations and revise reports through explicit versions rather than silently replacing the submitted record.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual GrantSubmission fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Required deliverables and approvals complete; submission receipt and revision history retained.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
