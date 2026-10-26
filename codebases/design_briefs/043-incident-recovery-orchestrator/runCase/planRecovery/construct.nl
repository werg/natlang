---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRecoveryStep
---
Construct a RecoveryStep draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Select a recovery path from approved runbooks using current evidence and legal action options. Sequence reversible actions first where appropriate, preserve prerequisites, and avoid improvising destructive commands or treating a successful command as recovered service.
