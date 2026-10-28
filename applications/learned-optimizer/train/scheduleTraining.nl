---
description: Choose the training work of the next cycle from the measured metrics.
args:
  metrics: TrainingMetrics
returns: TrainingSchedule
---
metrics.labelsStale is true after a bank publication: the supervised labels belong to the old bank. targetsAdmitted
counts supervised targets; proposalsRecorded counts the learner's own scored proposals; contrastGroups counts
proposal groups whose rewards differ, noContrastGroups those whose rewards are all equal; advantageSnr is the
signal-to-noise of those differences; ladder holds the latest median gaps.

1. When labelsStale is true, start with "refit-labels".
2. When targetsAdmitted is 0, return the work so far followed by "done", with reason "no admitted targets".
3. Add "supervised" with supervisedSteps 50. Supervised imitation gives dense learning while the optimizer is weak.
4. Add "collect-proposals". When contrastGroups is at least 8 and advantageSnr is at least 1, add "policy" after it:
   the reward signal is strong enough for the expected-score update. Otherwise the recorded proposals feed filtered
   imitation through the next refit.
5. When metrics.step is a positive multiple of 200, add "publish-bank" and then "ladder".
6. Return { work, supervisedSteps: 50, reason } where reason names the metric that decided each added entry.
