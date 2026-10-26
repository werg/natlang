---
args: { draft: DraftTaxonomyDefect, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated TaxonomyDefect field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Proposed structural changes are supported by multiple informative examples or a decisive new requirement. Explain the decisive evidence and smallest repair target; never weaken the gate.
