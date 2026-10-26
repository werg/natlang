---
description: Assemble reset-safe staff instructions and session-isolated operations.
args:{room:EscapeRoomState;proof:PuzzleVerification;variants:AccessibilityRoute[];playtests:PlaytestFinding[]}
returns:EscapeOperationsPacket
---
Do not mark ready while proof reports unreachable goals, cycles, missing props or failed exit invariants, or while a playtest is blocked. Bind setup and reset checklists to exact prop identifiers and room revision. Provide progressive hints and staff recovery instructions for each repairable scenario. Return setup and reset checklists, but leave checkpoint.resetVerified false because this operation has no post-reset readback. Include nextSessionId only when an authoritative session allocator supplied it; otherwise leave it absent. Preserve incident evidence without carrying solved flags between groups. Only verify a reset in the separate verifySessionReset operation when a current readback is supplied.
