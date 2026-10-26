---
args: { draft: DraftHandoverEvidence, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HandoverEvidence field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Artifact presence alone does not satisfy substantive evidence requirements. Explain the decisive evidence and smallest repair target; never weaken the gate.
