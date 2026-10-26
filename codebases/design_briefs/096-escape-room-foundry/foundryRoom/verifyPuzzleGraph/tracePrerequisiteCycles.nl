---
description: Identify prerequisite cycles and connect them to goals and explored failure states.
args: { room: EscapeRoomState; frontier: PuzzleFrontierAnalysis }
returns: CycleAssessment
---
Trace directed prerequisite edges from each node to each node in its `requires` list. Report only nodes on a real directed cycle, including a self-edge. For each cycle, call classifyCycleImpact(room, frontier, cycleNodeIds). A cycle among optional nodes is still reported, but do not call it a soft lock unless the frontier shows an incomplete terminal state or an unreachable required goal. Keep exact cycle node IDs and goal IDs; do not infer a cycle from shared clue wording.
