---
args: { draft: DraftReleaseReadiness, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to ReleaseReadiness: No release readiness claim over mismatched versions or unresolved blocking compatibility failures. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
