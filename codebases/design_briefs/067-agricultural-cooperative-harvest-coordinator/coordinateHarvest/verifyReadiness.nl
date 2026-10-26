---
description: Exact counterfactual gate for evidence-backed harvest readiness and quantity uncertainty.
args: { draft: DraftHarvestReadiness; prepared: PreparedEvidence; decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the HarvestReadiness gate to the supplied lot. Pass only when current evidence supports maturity, exact lot identity, quality findings, readiness window and stated uncertainty without upgrading an estimate into a guarantee. Fail when supplied evidence directly contradicts the draft or it treats known excluded quantity or capacity as available. Return needs-review when a decisive source, version, lot identity or authority is missing. Preserve the finite verdict exactly.
