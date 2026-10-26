---
args: { draft: DraftModerationDecision, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ModerationDecision field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact score bounds and consistent retrospective treatment under authorized moderation. Explain the decisive evidence and smallest repair target; never weaken the gate.
