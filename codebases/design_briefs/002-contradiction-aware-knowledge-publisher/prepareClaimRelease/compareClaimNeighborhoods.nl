---
args: {"claims": "QualifiedProposition[]", "desk": "PublicationDesk"}
returns: "PropositionComparison[]"
---
Group by supported subject identity and facet, then enumerate relevant pairs within those finite neighborhoods. Bind comparison = adjudicateScopedPair(left, right, desk.pages or empty) for each pair. Preserve coexisting populations and editions as such. A lagging translation follows its sourceEdition relation and becomes edition-mismatch; do not label it false merely because the current source changed. Return the smallest incompatible propositions and overlap for actual contradictions.
