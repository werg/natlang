---
args: { turn: ApprenticeshipTurn }
returns: InterpretedUtterance
---
Bind kind = classifyQuestionOrAction(turn.utterance). Preserve its exact kind, intent, and target in the returned `kind` field; do not independently recategorize it. For a question, bind answer = resolveFromVisibleActorFacts(turn.utterance, turn.actor, turn.world), then return { kind, answer, clarificationNeeded: [] }. For a clear action proposal, return { kind, clarificationNeeded: [] } without answering or changing the world. For an ambiguous utterance, return { kind, clarificationNeeded: ["Clarify whether this is a question or a proposed operation, and identify its intended target."] }; do not create an executable action.
