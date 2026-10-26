---
args: { draft: DraftTestPartition, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TestPartition field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Partitions cover each requirement’s decisive conditions and failure behavior. Explain the decisive evidence and smallest repair target; never weaken the gate.
