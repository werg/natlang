---
args: { goal: GoalContract, policy: PolicySelection, candidates: GrantCondition[], evidence: EvidencePage[] }
returns: StageResultGrantAllocation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Match expenditures to eligible activities using descriptions, service periods, and supporting records. Split shared costs only under documented allocation rules, detect duplicate funding attribution, and retain unsupported portions for review.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual GrantAllocation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact amount conservation and no double attribution across grants.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
