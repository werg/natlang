---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DisruptionImpact[] }
returns: StageResultShippingCommitment
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile order terms, customer messages, and product constraints to identify what must be preserved. Separate contractual deadlines from preferences, recognize dependencies such as installation appointments, and avoid ranking customers by unstated commercial assumptions. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultShippingCommitment with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every priority follows the supplied policy and actual commitment evidence.
