---
args: { action: TypedTrainingAction, procedure: ProcedureStep }
returns: InvariantCheck
---
Confirm that the action's prerequisite observations were recorded and the operation is allowed at this procedure step. If the target is prohibited or a safety condition is unmet, require the training refusal branch. Preserve the prior state when a prerequisite check fails.
