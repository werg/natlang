---
args: { draft: DraftGardenPreference, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated GardenPreference field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Member commitments are explicit and attributable. Explain the decisive evidence and smallest repair target; never weaken the gate.
