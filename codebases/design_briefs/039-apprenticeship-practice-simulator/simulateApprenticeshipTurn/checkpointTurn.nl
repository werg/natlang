---
args: { prior: PracticeCheckpoint, receipt: ApprenticeshipTurnReceipt }
returns: PracticeCheckpoint
---
Append this complete simulation receipt once, keyed by receipt.binding.turnId. It must retain the original bound input, returned interpretation, matched typed action, simulator transition, challenge and debrief. Append the turn ID to appliedTransitionIds, the challenge ID to seenChallengeIds, set lastTransitionId and worldStateJson from the exact returned transition, and increment the checkpoint version. Preserve the actor knowledge partition unless an explicit transition event updates it. Never overwrite a prior receipt; a duplicate ID with missing or changed receipt is a conflict to resolve before this function is called.
