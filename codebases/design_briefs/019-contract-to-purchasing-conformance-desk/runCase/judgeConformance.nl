---
args: { goal: GoalContract, policy: PolicySelection, candidates: AgreementScope[], evidence: EvidencePage[] }
returns: StageResultConformanceFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare each requested line with contract inclusions, exclusions, and conditions. Separate commercial price compliance from functional scope and preserve conditions such as minimum quantities, required bundles, or approved use limitations.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ConformanceFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: A catalog listing alone does not establish contractual coverage.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
