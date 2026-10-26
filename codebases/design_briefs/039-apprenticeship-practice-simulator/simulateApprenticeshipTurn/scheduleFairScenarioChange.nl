---
args: { transition: PracticeWorldTransition, turn: ApprenticeshipTurn, sourceScenario: PracticeScenario }
returns: SeededPracticeChallenge
---
Bind candidate = chooseUnseenChallenge(turn.scenarioFamily, turn.learningObjective, turn.seed, turn.checkpoint.seenChallengeIds). Test that the learner has reached the prerequisite decision point and that the new challenge changes a relevant choice. Bind fairness = testRecoverableWithVisibleFacts(candidate, transition.world). If no safe recovery route exists, defer the challenge. Record the seed, candidate ID, and visible-fact snapshot so replay is reproducible.
