---
args: { draft: DraftAssessmentItem, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to AssessmentItem: Item is answerable from supplied knowledge and permitted resources. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
