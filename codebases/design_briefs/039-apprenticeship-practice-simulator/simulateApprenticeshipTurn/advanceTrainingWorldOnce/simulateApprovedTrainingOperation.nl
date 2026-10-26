---
args: { action: TypedTrainingAction, world: TrainingWorld, procedure: ProcedureStep }
returns: PracticeWorldTransition
---
Check action kind and target against the current procedure step. Apply only the declared simulator delta. If the action is unsafe, preserve world resources and return the refusal or approved simulated consequence. Bind resourceCheck = verifyTrainingResourceInvariant(world, candidateWorld), then bind procedureCheck = verifyProcedureTransition(action, procedure). Reject any candidate transition that violates either invariant; return the failed check and leave the prior state intact.
