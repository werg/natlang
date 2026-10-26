---
args: { goal: GoalContract, policy: PolicySelection, candidates: SamplingPlan[], evidence: EvidencePage[] }
returns: StageResultCodeDefinition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Draft codes from pilot material and research concepts, specifying boundaries through positive and negative examples. Separate descriptive labels from inferred explanations; reconcile overlapping codes without forcing genuinely multidimensional evidence into one category.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CodeDefinition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each code is usable on unseen examples and has a discriminating exclusion rule.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
