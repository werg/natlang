---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentedPromise[] }
returns: StageResultBehaviorObservation
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Form bounded sandbox probes using approved capabilities and representative inputs. Record environment and release identity, compare observed outcomes with the documented promise, and avoid treating one failed environment as proof of a product defect. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultBehaviorObservation with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Probe is reproducible and isolates the behavior under review.
