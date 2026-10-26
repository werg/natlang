---
args: { action: TypedTrainingAction, turn: ApprenticeshipTurn }
returns: PracticeWorldTransition
---
Bind replay = inspectTurnIdentifier(action, turn.checkpoint). If already applied, return its recorded transition unchanged. Otherwise bind candidate = simulateApprovedTrainingOperation(action, turn.world, turn.procedure). Verify resource and procedure invariants before saving it. Keep each simulated actor's knowledge partition unchanged unless the action's explicit information event permits disclosure. Unsafe proposals use the training refusal path and never call an external tool.
