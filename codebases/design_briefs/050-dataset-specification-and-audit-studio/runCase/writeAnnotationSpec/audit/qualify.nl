---
args: { draft: DraftAnnotationRule, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to AnnotationRule: Labels are grounded in available evidence and ambiguous cases have an explicit disposition. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
