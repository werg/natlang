---
args: { draft: DraftRepairDiagnostic, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RepairDiagnostic field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Diagnostic is approved, informative, and within assigned capability. Explain the decisive evidence and smallest repair target; never weaken the gate.
