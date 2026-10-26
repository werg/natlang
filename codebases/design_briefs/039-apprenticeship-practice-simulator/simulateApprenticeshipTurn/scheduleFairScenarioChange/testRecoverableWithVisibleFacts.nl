---
args: { candidate: ChallengeCandidate, world: TrainingWorld }
returns: ChallengeFairnessCheck
---
Verify that every action needed to recover from the complication is discoverable from currently visible facts or from a permitted question. Confirm that the challenge does not negate an earlier valid action. Return the required recovery sequence and any hidden prerequisite; defer any candidate that requires knowledge the learner could not acquire.
