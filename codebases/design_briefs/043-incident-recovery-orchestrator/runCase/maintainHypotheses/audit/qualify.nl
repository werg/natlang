---
args: { draft: DraftIncidentHypothesis, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to IncidentHypothesis: Hypotheses remain provisional until diagnostic evidence supports them. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
