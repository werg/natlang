---
args: { goal: GoalContract, policy: PolicySelection, candidates: Handover[], evidence: EvidencePage[] }
returns: StageResultCommitment
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read threads with their replies and amendments. Distinguish a tentative intention from a promise, identify the beneficiary and deliverable, and retain conditional commitments whose activation event has not yet occurred.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual Commitment fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every commitment has an attributable statement; missing dates stay unknown rather than invented.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
