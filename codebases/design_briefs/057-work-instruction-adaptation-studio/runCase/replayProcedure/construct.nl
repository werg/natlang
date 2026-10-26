---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftProcedureScenario
---
Construct a ProcedureScenario draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Generate ordinary and adverse scenarios involving local exceptions, missing resources, and interrupted work. Replay the adapted procedure to detect ambiguous instructions, skipped prerequisites, or branches that cannot reach a safe terminal state.
