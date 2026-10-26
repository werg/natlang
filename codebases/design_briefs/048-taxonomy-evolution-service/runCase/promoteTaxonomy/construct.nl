---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftTaxonomyRelease
---
Construct a TaxonomyRelease draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Present impact, examples, and reporting consequences to the steward, then publish a versioned release and migration plan. Notify dependent systems, invalidate only relevant classification caches, and preserve old definitions for historical queries.
