---
args: { draft: DraftCuratedObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to CuratedObservation: Rare but supported observations are retained; sensitive locations are appropriately generalized. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
