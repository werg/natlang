---
args: { prior: PracticeCheckpoint, transition: PracticeWorldTransition, challenge: SeededPracticeChallenge }
returns: PracticeCheckpoint
---
Append the completed turn ID, transition reference, revealed-fact references, and challenge seed to the versioned session checkpoint. Preserve the original world state and actor knowledge partition for audit. A retry with the same turn ID must recover this result rather than create another transition.
