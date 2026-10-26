---
args: { draft: DraftRepairSymptom, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RepairSymptom field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Observations and inferred causes remain distinct; risk flags trigger specialist routing. Explain the decisive evidence and smallest repair target; never weaken the gate.
