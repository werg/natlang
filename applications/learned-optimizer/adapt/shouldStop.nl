---
description: Whether the adaptation episode should stop now.
args:
  state: AdaptationState
returns: boolean
---
1. Return true when state.remainingSteps is 0.
2. Return true when the last three entries of state.history were all rejected: further revisions are not paying off.
3. Return true when the last entry's operator was "keep".
4. Otherwise return false.
