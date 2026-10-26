---
description: Decide whether one exact failure branch has a safe recovery that preserves the required puzzle outcome.
args: { room: EscapeRoomState; proof: PuzzleVerification; variants: AccessibilityRoute[]; sourceEvidence: EvidenceItem[]; case: FailureCase }
returns: PlaytestFinding
---
1. Confirm case.stateId exists in proof.states. If it does not, return block and name the mismatched state; do not generalize from another branch.
2. For lost-prop, determine whether a currently enabled alternate path reaches the same required goal without that prop. If none exists, accept only a source-supported staff recovery that restores the puzzle state or substitutes an approved equivalent; a proposed replacement alone is repair-required.
3. For repeat-action, require an idempotent result: applying the same action twice cannot consume another resource, duplicate a required clue or invalidate a solved state. Keep the action unresolved when simulator behavior is not evidenced.
4. For out-of-order discovery, check whether the early clue remains interpretable without designer-only context and whether its dependent action is still recoverable. Preserve any ambiguity that creates a false solve or blocks a later reveal.
5. For accessibility, trace the route's equivalentClueIds to the same terminalNodeId and decision information. A route label without equivalent clue support is not a passing path.
6. Return the supplied case kind and IDs, a specific disposition (`safe-recovery`, `repair` or `block`) and the actual staffed recovery step or missing condition. The exit remains independent of all puzzle recovery.
