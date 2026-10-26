---
args: { draft: DraftChangeWorkPackage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to ChangeWorkPackage: Feasible version transition and explicit handling of existing stock and in-flight work. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
