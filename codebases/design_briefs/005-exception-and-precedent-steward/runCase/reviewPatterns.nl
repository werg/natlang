---
args: { goal: GoalContract, policy: PolicySelection, candidates: ExceptionDisposition[], evidence: EvidencePage[] }
returns: StageResultPolicyReviewSignal
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Group substantively similar exceptions while discounting copied requests and a single repeated incident. Formulate a policy revision and boundary examples, then simulate historical outcomes before referring the proposal to the policy owner.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PolicyReviewSignal fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Frequency is not automatic justification; proposed revision passes supplied counterexample cases.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
