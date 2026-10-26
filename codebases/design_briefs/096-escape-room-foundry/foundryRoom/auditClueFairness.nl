---
description: Review clue sufficiency and intended reasoning over verified reachable puzzle states.
args:{room:EscapeRoomState;proof:PuzzleVerification}
returns:{ambiguousClueIds:string[];lateClueDependencies:string[];misreadingCandidates:string[];repairTargets:string[];}
---
For each required transition, identify which visible clue communicates the needed information and whether the clue becomes discoverable before the decision. Check out-of-order discovery for contradictions. Name reasonable alternate readings and retain both acceptable answers when the puzzle design supports them; otherwise revise the artifact rather than adding unexplained hints.
