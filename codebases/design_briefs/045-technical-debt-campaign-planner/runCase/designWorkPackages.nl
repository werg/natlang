---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtMechanism[] }
returns: StageResultDebtWorkPackage
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Convert supported mechanisms into finite improvement packages with explicit interfaces and non-goals. Propose alternatives with different disruption levels, avoid unbounded rewrites, and define completion through changed behavior rather than merged code alone. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultDebtWorkPackage with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Each package is executable and tied to the original symptom or validated mechanism.
