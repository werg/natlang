---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: StageResultProcedureRequirement
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Parse the master instruction into actions, guards, prerequisites, and verification obligations. Distinguish explanatory advice from mandatory controls, preserve warnings in their operative context, and identify cross-references needed to execute the procedure correctly. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultProcedureRequirement with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every mandatory control has a traceable source and cannot disappear during simplification.
