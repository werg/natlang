---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: BehavioralRequirement[] }
returns: StageResultTestPartition
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Partition input and state space by behaviorally meaningful distinctions, including invalid inputs and interacting requirements. Identify boundary cases and previously untested combinations rather than generating many superficial variants of the same happy path. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultTestPartition with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Partitions cover each requirement’s decisive conditions and failure behavior.
