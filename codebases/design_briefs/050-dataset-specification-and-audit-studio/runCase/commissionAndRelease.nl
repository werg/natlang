---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CoverageFinding[] }
returns: StageResultDatasetRelease
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Create targeted acquisition or annotation tasks, rerun audits on new material, and assemble a release manifest with known limits. Freeze splits by source and world family, prevent leakage, and require owner approval for the specific release use. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDatasetRelease with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Rights, quality, split integrity, and required coverage gates pass or limitations block release.
