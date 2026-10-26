---
args: { draft: DraftTaxonomyImpact, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TaxonomyImpact field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact changed-record set and no fabricated historical precision. Explain the decisive evidence and smallest repair target; never weaken the gate.
