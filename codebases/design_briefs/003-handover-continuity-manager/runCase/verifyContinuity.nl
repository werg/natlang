---
args: { goal: GoalContract, policy: PolicySelection, candidates: OwnershipTransfer[], evidence: EvidencePage[] }
returns: StageResultContinuityReview
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Revisit the portfolio after departure, follow missing acknowledgments, and compare completed artifacts to promises. Close the handover only when ownership and immediate next steps are established, while operational work may legitimately remain open.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ContinuityReview fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Handover completion and task completion are separate statuses with separate gates.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
