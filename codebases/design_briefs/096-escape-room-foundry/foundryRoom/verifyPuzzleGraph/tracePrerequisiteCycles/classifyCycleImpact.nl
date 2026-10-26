---
description: Distinguish a prerequisite cycle from the goal and state failures it actually causes.
args: { room: EscapeRoomState; frontier: PuzzleFrontierAnalysis; cycleNodeIds: string[] }
returns: { blockedGoals: string[]; affectedTerminalStateIds: string[]; finding: string }
---
For each required goal, trace its prerequisite chain through the exact node IDs. Include a goal in blockedGoals only when every route to solving it depends on a node in the cycle. Preserve an alternative route when an acyclic solution exists. Match affected terminalStateIds only to frontier states whose enabled node list is empty while one of those dependent goals remains unsolved. Explain the cycle's actual effect; a cycle alone does not prove every room branch is locked.
