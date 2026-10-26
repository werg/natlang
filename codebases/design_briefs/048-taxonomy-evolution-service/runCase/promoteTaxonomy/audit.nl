---
args: { draft: DraftTaxonomyRelease, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TaxonomyRelease field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Approved definitions, migration scope, and dependent consumer compatibility are verified. Explain the decisive evidence and smallest repair target; never weaken the gate.
