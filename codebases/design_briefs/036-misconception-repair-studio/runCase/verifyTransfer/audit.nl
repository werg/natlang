---
args: { draft: DraftTransferFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TransferFinding field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Mastery requires novel unassisted evidence under the supplied competency standard. Explain the decisive evidence and smallest repair target; never weaken the gate.
