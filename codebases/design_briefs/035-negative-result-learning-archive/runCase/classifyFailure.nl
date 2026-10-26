---
args: { goal: GoalContract, policy: PolicySelection, candidates: ResearchAttempt[], evidence: EvidencePage[] }
returns: StageResultNegativeFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Determine whether evidence supports a negative result, an invalid measurement, an unfinished procedure, or genuine uncertainty. Preserve the distinction even when all cases were informally described as failures and avoid stronger conclusions than the attempt warrants.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual NegativeFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: A failed process cannot be treated as evidence against the hypothesis it never validly tested.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
