---
args: { draft: DraftProcedureRequirement, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to ProcedureRequirement: Every mandatory control has a traceable source and cannot disappear during simplification. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
