---
args: { goal: GoalContract, policy: PolicySelection, candidates: PolicyCaseFacts[], evidence: EvidencePage[] }
returns: StageResultOutcomeDifference
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Execute both policy versions with isolated state and compare duties, permissions, timing, and escalation. Trace differences to specific changed conditions and separate genuine policy effects from nondeterministic or unsupported semantic judgments.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OutcomeDifference fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Paired runs share facts and executor settings; unexplained differences remain diagnostic failures.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
