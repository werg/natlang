---
args: { goal: GoalContract, policy: PolicySelection, candidates: BoundaryScenario[], evidence: EvidencePage[] }
returns: StageResultPolicyOption
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Summarize who and what changes under each candidate, then propose clarifications or alternatives for the policy owner. Preserve minority and exceptional cases, report coverage limits, and avoid choosing normative priorities not supplied by the organization.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PolicyOption fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Policy owner sees exact changed-case sets and unresolved judgments before adoption.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
