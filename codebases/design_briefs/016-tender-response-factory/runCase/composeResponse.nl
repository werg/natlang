---
args: { goal: GoalContract, policy: PolicySelection, candidates: ContributionTask[], evidence: EvidencePage[] }
returns: StageResultResponseSection
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Draft consistent answers using supported claims and approved commercial positions. Tailor explanations to the customer’s evaluation criteria without inventing facts; preserve declared exceptions and reconcile promises across technical, pricing, and delivery sections.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResponseSection fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each material claim has evidence or a clearly labeled future commitment approved by its owner.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
