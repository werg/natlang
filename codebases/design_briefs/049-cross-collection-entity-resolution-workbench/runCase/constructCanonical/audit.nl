---
args: { draft: DraftCanonicalEntity, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CanonicalEntity field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Canonical values remain attributable and do not erase meaningful disagreement. Explain the decisive evidence and smallest repair target; never weaken the gate.
