---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftEnvironmentalAllocation
---
Construct a EnvironmentalAllocation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Allocate shared activities using supplied rules and exact calculations, checking for duplicate records or overlapping contractor reports. Distinguish a transferred responsibility from a duplicated physical activity and preserve unresolved attribution rather than balancing by assumption.
