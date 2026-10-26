---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ResponseObservation[] }
returns: StageResultMisconceptionHypothesis
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Generate a small set of plausible explanations for the observed pattern, including task ambiguity and accidental error. Identify a question whose answers would distinguish these explanations instead of treating one wrong response as a stable misconception. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultMisconceptionHypothesis with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: No diagnosis based on identity or a single uninformative response.
