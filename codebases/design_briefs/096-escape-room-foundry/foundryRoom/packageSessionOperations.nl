---
description: Assemble reset-safe staff instructions and session-isolated operations.
args:{room:EscapeRoomState;proof:PuzzleProof;variants:AccessibilityRoute[];playtests:PlaytestFinding[]}
returns:EscapeOperationsPacket
---
Do not mark ready while proof reports unreachable goals, cycles, missing props or failed exit invariants, or while a playtest is blocked. Bind setup and reset checklists to exact prop identifiers and room revision. Provide progressive hints and staff recovery instructions for each repairable scenario. Verify a reset against the declared baseline and new session ID; preserve incident evidence without carrying solved flags between groups.
