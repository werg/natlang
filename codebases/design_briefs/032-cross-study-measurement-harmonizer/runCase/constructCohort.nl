---
args: { goal: GoalContract, policy: PolicySelection, candidates: HarmonizationRule[], evidence: EvidencePage[] }
returns: StageResultHarmonizedSlice
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Define comparable cohorts before materialization, apply approved transformations in a sandbox, and reconcile row counts and exclusions. Keep missingness reasons and study membership available rather than flattening them into one generic null category.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual HarmonizedSlice fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact row lineage, unit consistency, and no unsupported cross-study comparability.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
