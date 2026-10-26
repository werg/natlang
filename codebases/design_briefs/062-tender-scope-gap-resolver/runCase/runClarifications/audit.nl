---
args: { draft: DraftScopeClarification, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScopeClarification field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: A clarification closes a gap only with explicit scoped acceptance. Explain the decisive evidence and smallest repair target; never weaken the gate.
