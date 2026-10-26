---
args: { goal: GoalContract, policy: PolicySelection, candidates: ObligationRelation[], evidence: EvidencePage[] }
returns: StageResultTriggerFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare incoming operational events with contractual triggers and conditions. Distinguish an allegation that a trigger occurred from established evidence, and request the smallest missing fact when activation would materially change a duty or deadline.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual TriggerFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Trigger activation requires evidence under the selected interpretation version.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
