---
args: { goal: GoalContract, policy: PolicySelection, candidates: ObligationTask[], evidence: EvidencePage[] }
returns: StageResultObligationStatus
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare completed actions with the actual duty and any acceptance condition. Distinguish performance from waiver or settlement, record reviewer-approved interpretations, and keep disputed discharge visible rather than closing on an internal completion flag.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ObligationStatus fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Satisfaction is supported; waivers require their own scoped authority evidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
