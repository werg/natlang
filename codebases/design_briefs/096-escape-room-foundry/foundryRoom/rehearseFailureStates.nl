---
description: Exercise lost props, repeated actions and soft-lock recovery with staff procedures.
args:{room:EscapeRoomState;proof:PuzzleProof;variants:AccessibilityRoute[]}
returns:PlaytestFinding[]
---
For every reachable state, test missing and broken props, repeated solve input, early goal discovery and opt-out route. A broken prop must have a documented recovery that restores the intended state without revealing designer-only facts. If no safe exit or staff path exists, block the design rather than treating a hint as recovery.
