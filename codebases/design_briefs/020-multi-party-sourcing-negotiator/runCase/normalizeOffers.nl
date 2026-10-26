---
args: { goal: GoalContract, policy: PolicySelection, candidates: SourcingRequirement[], evidence: EvidencePage[] }
returns: StageResultSupplierOffer
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconcile quotations into comparable units and scopes while preserving bundles, exclusions, and conditional discounts. Distinguish an indicative quote from a binding offer and expose missing service or logistics costs rather than assuming comparability.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SupplierOffer fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Exact totals and quote validity; no hidden removal of supplier conditions.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
