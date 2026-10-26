---
args: { draft: DraftHarvestRevision, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated HarvestRevision field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Revised commitments are accepted and actual quantities remain traceable by lot. Explain the decisive evidence and smallest repair target; never weaken the gate.
