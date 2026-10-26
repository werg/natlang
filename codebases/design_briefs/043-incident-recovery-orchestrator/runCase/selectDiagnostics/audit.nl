---
args: { draft: DraftDiagnosticAction, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DiagnosticAction field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Diagnostic action is allowlisted and cannot widen credentials or query scope. Explain the decisive evidence and smallest repair target; never weaken the gate.
