---
args: { goal: GoalContract, policy: PolicySelection, candidates: DeliveryEvidence[], evidence: EvidencePage[] }
returns: StageResultAcceptanceFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Interpret customer responses under the agreement’s acceptance process. Separate acknowledgment of receipt from acceptance, verify the respondent’s authority, and apply deemed-acceptance timing only when all supplied contractual prerequisites are established.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual AcceptanceFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No implied acceptance outside the configured agreement; exact date calculations use host services.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
