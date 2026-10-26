---
args: { draft: DraftScopeCoverage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ScopeCoverage field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Coverage is phase-specific and evidence-backed. Explain the decisive evidence and smallest repair target; never weaken the gate.
