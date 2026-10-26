---
args: { variant: WorkfloorInstructionDraft, restarts: RestartCondition[], exceptions: LocalException[] }
returns: ReplayFinding
---
Bind invariants = verifyRestartAndGuardCoverage(variant.restartProtocols, variant.inspectionGuards, restarts, exceptions). Replay each interruption from its recorded state, requiring every isolation and repeat check before resuming. Report exact failed steps and preserve inspection guards under each exception; incomplete restart or a missing guard blocks release.
