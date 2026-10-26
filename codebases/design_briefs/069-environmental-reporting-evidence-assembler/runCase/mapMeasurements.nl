---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReportingBoundary[] }
returns: StageResultEnvironmentalMeasure
---
1. Prepare a bounded evidence slice: `prepared = await prepare(input, goal, policy, evidencePlan, previous)`; the slice includes source versions, contrary facts and missing fields.
2. Apply this stage algorithm: Reconcile measurements, invoices, contractor statements, and field records with their actual methods and periods. Preserve estimates versus direct measurements, identify missing coverage, and retain original units and uncertainty for exact conversion services. Obtain `decisions = await decide(prepared, goal, policy)` and construct `draft = await construct(decisions, prepared, input)`.
3. Audit `draft` through `audit(draft, prepared, decisions)`. Only passing records are accepted; retain each unresolved or rejected candidate in coverage.
4. Return StageResultEnvironmentalMeasure with accepted records, unresolved drafts, consumed refs, coverage, questions, proposed actions and checks. Domain gate: Every value has method, period, unit, and provenance.
