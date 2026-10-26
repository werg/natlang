---
args: { draft: DraftTestFailureFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to TestFailureFinding: Failure classification is evidence-backed and environment errors are not counted as product coverage. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
