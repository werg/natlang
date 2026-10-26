---
description: Return a complete held operations packet when the recomputed proof or its supplied corroboration is invalid.
args: { room: EscapeRoomState; proof: PuzzleVerification; suppliedCheck: PuzzleProofCheck }
returns: EscapeOperationsPacket
---
Return the complete packet with this exact room and recomputed proof, no playtest or operational checklist represented as complete, and checkpoint { roomId: room.id, revision: room.revision, resetVerified: false }. Leave nextSessionId absent because this call has no allocator receipt. Set status blocked. Preserve the proof's structural references, duplicate nodes, property gaps, unreachable goals, cycles, frontier lock states and egress review; set blockers to every suppliedCheck difference plus the exact structural, reachability, cycle, prop and exit findings that make this graph invalid. Do not copy the caller's proof into the output.
