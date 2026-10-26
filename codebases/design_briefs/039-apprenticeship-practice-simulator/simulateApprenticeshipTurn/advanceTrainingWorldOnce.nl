---
args: { action: TypedTrainingAction, turn: ApprenticeshipTurn }
returns: PracticeWorldTransition
---
Bind candidate = simulateApprovedTrainingOperation(action, turn.world, turn.procedure). This operation receives only a newly classified action after the public root has ruled out replay and clarification. Verify resource and procedure invariants before saving the transition. Keep each simulated actor's knowledge partition unchanged unless the action's explicit information event permits disclosure. Unsafe proposals use the training refusal path and never call an external tool.
