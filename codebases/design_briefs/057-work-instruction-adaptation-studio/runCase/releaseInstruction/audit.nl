---
args: { draft: DraftInstructionRelease, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated InstructionRelease field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Approved version and site applicability are visible at every point of use. Explain the decisive evidence and smallest repair target; never weaken the gate.
