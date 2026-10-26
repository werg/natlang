---
args: { utterance: InterpretedUtterance, turn: ApprenticeshipTurn }
returns: TypedTrainingAction
---
Compare the proposed operation with the scenario's allowlisted action contract and current procedure step. Match only when target, preconditions, and learner intent are explicit. If intent or a consequence-bearing detail is missing, return status clarification and put the precise question(s) in clarificationNeeded; do not route it to world advancement. For an executable or refusal action return clarificationNeeded: []. A useful unexpected question may update the learner's information state but is not itself a world mutation. An unsafe proposal is typed as `refuse-in-training` with a cited policy rule.
