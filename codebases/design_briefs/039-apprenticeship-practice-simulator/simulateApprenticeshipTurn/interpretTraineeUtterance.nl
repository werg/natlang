---
args: { turn: ApprenticeshipTurn }
returns: InterpretedUtterance
---
Bind kind = classifyQuestionOrAction(turn.utterance). A question asks for information; an imperative or stated intent proposes a typed operation. If the wording can alter a valve, record, or safety condition in more than one way, return clarificationNeeded and no executable action. For a question, ask resolveFromVisibleActorFacts before answering; scripted prompt matching is not a prerequisite.
