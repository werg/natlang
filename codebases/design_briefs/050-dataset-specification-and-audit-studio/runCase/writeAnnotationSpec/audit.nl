---
args: { draft: DraftAnnotationRule, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated AnnotationRule field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Labels are grounded in available evidence and ambiguous cases have an explicit disposition. Explain the decisive evidence and smallest repair target; never weaken the gate.
