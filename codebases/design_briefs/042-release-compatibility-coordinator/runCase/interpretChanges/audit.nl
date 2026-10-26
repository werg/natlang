---
args: { draft: DraftBehavioralChange, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated BehavioralChange field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every claimed behavioral change has evidence or a targeted verification task. Explain the decisive evidence and smallest repair target; never weaken the gate.
