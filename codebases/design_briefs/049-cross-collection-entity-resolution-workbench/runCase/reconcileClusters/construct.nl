---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIdentityCluster
---
Construct a IdentityCluster draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Build clusters using constrained graph operations rather than unconditional transitive closure. Before merging components, test all relevant cannot-link and temporal constraints; route inconsistent triangles to review and preserve uncertain alternatives.
