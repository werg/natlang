---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultIdentityCandidate
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Generate candidate pairs using exact normalized keys, aliases, and bounded semantic retrieval. Preserve blocking provenance and include audit samples outside candidate blocks so recall failures are measurable rather than hidden by the candidate generator. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultIdentityCandidate with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Candidate generation cannot cross access boundaries or omit known hard incompatibilities.
