---
description: Migration stopping judgment. Whether the repair loop should stop - the checks pass, the budget is spent, nothing left is repairable, or the work repeats.
args:
  state: RepairState
returns: boolean
---
Decide whether the repair loop stops at state. Answer true when any of these holds, exactly in eval:

1. state.validation is not null and its status is "passed".
2. state.remaining is 0 (the budget of rounds is spent).
3. state.findings is not empty and none of them is repairable (the rest is environment or unrelated).
4. state.revisions holds the same revision twice (the repair came back to a candidate it already tried).

Otherwise answer false. A candidate whose checks fail with repairable findings and budget left is not settled.
