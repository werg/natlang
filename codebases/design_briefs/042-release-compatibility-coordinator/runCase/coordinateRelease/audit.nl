---
args: { draft: DraftReleaseReadiness, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReleaseReadiness field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No release readiness claim over mismatched versions or unresolved blocking compatibility failures. Explain the decisive evidence and smallest repair target; never weaken the gate.
