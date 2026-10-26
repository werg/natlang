---
args: { draft: DraftDiagnosticAction, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to DiagnosticAction: Diagnostic action is allowlisted and cannot widen credentials or query scope. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
