---
args: { draft: DraftPlotConstraint, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PlotConstraint field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Planning uses verified or explicitly uncertain conditions, not invented agronomic facts. Explain the decisive evidence and smallest repair target; never weaken the gate.
