---
args: { turn: ApprenticeshipTurn }
returns: ApprenticeshipTurnResult
---
Bind utterance = interpretTraineeUtterance(turn). If it is a question, answer only from the actor's visible world facts and reveal no hidden fact. If it is an action proposal, bind typedAction = matchPermittedProcedureStep(utterance, turn). Ambiguous requests remain unexecuted until clarified.
Bind transition = advanceTrainingWorldOnce(typedAction, turn). The turn identifier is an idempotency key: if it already appears in the saved checkpoint, return its prior transition and do not apply its resource delta again. Unsafe proposals receive only an in-simulator refusal or approved training consequence; nothing dispatches to real equipment.
Bind challenge = scheduleFairScenarioChange(transition, turn, turn.sourceScenario). Derive the learning target from the preserved source scenario and saved seed while varying only surface context; retain earlier facts so a correct previous decision cannot become retroactively wrong.
Bind debrief = compareObservedProcedure(turn.actionHistory, transition, challenge). Separate a lucky outcome from correct sequence, checks, and escalation behavior. Cite each conclusion to a recorded learner action.
Return { sessionId: turn.sessionId, transition, challenge, debrief, checkpoint: checkpointTurn(turn.checkpoint, transition, challenge), commitAllowed: false }.
