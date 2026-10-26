---
args: { baseline: ProcedureRequirement, ambiguity: AmbiguitySet, warnings: WarningIntegrity, exceptions: LocalException[], restarts: RestartCondition[] }
returns: WorkfloorInstructionDraft
---
Bind mappedControls = mapProcedureStepsToVerifiedControls(baseline, ambiguity). Use mappedControls for the variant local step identifiers. Carry all baseline safety controls forward. Apply a local exception only to its stated step and constraint, leaving listed inspections as mandatory. Attach translated warning text only when its warning integrity is safe. Write restart instructions from the exact interruption state and required rechecks.
