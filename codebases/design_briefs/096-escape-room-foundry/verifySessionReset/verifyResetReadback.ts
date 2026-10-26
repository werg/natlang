import type { EscapeOperationsCheckpoint, EscapeRoomState, ResetReadback } from "../types.js";

export type ResetVerification = { verified: boolean; findings: string[] };

export default function verifyResetReadback(
  room: EscapeRoomState,
  checkpoint: EscapeOperationsCheckpoint,
  readback: ResetReadback | null,
): ResetVerification {
  if (!readback) return { verified: false, findings: ["no_reset_readback"] };

  const findings: string[] = [];
  if (checkpoint.roomId !== room.id) findings.push("checkpoint_room_mismatch");
  if (checkpoint.revision !== room.revision) findings.push("checkpoint_revision_mismatch");
  if (readback.roomId !== room.id) findings.push("room_identity_mismatch");
  if (readback.roomRevision !== room.revision) findings.push("room_revision_mismatch");
  if (!checkpoint.nextSessionId || readback.sessionId !== checkpoint.nextSessionId) findings.push("session_identity_mismatch");
  if (!readback.receiptId.trim() || !readback.observedAt.trim()) findings.push("readback_receipt_incomplete");
  if (!readback.solvedFlagsCleared) findings.push("solved_flags_not_cleared");
  if (!readback.exitAvailable || !room.exitAlwaysAvailable) findings.push("emergency_exit_not_verified");

  const expected = new Map(room.props.map(prop => [prop.id, prop]));
  const observed = new Map<string, number>();
  for (const item of readback.propObservations) {
    observed.set(item.propId, (observed.get(item.propId) ?? 0) + 1);
    const baseline = expected.get(item.propId);
    if (!baseline) findings.push(`unexpected_prop:${item.propId}`);
    else {
      if (item.present !== baseline.present) findings.push(`presence_mismatch:${item.propId}`);
      if (item.working !== baseline.working) findings.push(`condition_mismatch:${item.propId}`);
      if (item.sessionId !== checkpoint.nextSessionId) findings.push(`prop_session_mismatch:${item.propId}`);
    }
  }
  for (const id of expected.keys()) {
    const count = observed.get(id) ?? 0;
    if (count === 0) findings.push(`missing_prop_observation:${id}`);
    if (count > 1) findings.push(`duplicate_prop_observation:${id}`);
  }
  return { verified: findings.length === 0, findings };
}
