---
args: { draft: DraftDiagnosticTask, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DiagnosticTask field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Diagnostic outcomes discriminate hypotheses and do not introduce unlearned prerequisites. Explain the decisive evidence and smallest repair target; never weaken the gate.
