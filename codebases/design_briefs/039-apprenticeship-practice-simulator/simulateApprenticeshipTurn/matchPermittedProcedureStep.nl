---
args: { utterance: InterpretedUtterance, turn: ApprenticeshipTurn }
returns: TypedTrainingAction
---
Compare the proposed operation with the scenario's allowlisted action contract and current procedure step. Match only when target, preconditions, and learner intent are explicit. A useful unexpected question may update the learner's information state but is not itself a world mutation. An unsafe proposal is typed as `refuse-in-training` with a cited policy rule.
