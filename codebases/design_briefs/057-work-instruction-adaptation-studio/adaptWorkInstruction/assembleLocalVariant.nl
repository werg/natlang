---
args: { baseline: ProcedureRequirement, ambiguity: AmbiguitySet, warnings: WarningIntegrity, exceptions: LocalException[], restarts: RestartCondition[] }
returns: WorkfloorInstructionDraft
---
Bind mappedControls = mapProcedureStepsToVerifiedControls(baseline, ambiguity). Bind guards = preserveInspectionGuardMap(baseline, exceptions, mappedControls). Bind restartPlans = buildRestartProtocols(restarts, mappedControls, guards). Bind stepWarnings = bindWarningAssertionsToSteps(warnings.stepBindings, mappedControls).
Use mappedControls as exact local step identifiers. Carry every baseline safety control into guards; permit a waiver only for its named step and constraint while preserving each inspection explicitly. Attach warning language only to steps named by its evidence and omit unsafe translations. Each restart protocol starts from its recorded interruption state and carries the required isolation and repeat checks.
