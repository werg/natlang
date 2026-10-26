---
description: Exact counterfactual gate for preservation of an observation's source facts and uncertainty.
args: { draft: DraftFieldObservation; prepared: PreparedEvidence; decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Pass only when observer, observedAt, locationRef, description and method remain supported by current evidence and original uncertainty is preserved. Fail when the draft invents precision, changes the original description, or asserts an unsupported time or method. Return needs-review when a decisive metadata field, source version, or authority is absent. Return the finite verdict only.
