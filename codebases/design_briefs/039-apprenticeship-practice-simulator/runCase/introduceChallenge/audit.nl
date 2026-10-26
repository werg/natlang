---
args: { draft: DraftPracticeComplication, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated PracticeComplication field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Complication is fair, procedurally relevant, and reproducible from the scenario seed. Explain the decisive evidence and smallest repair target; never weaken the gate.
