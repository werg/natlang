import type { ActionCheck, LegalAction, PersistentWorldState, PlayerUtterance } from "../../types.js";

export default function validateActionEnvelope(
  world: PersistentWorldState,
  utterance: PlayerUtterance,
  action: LegalAction,
): ActionCheck {
  const blockers: string[] = [];
  if (!action.id.trim()) blockers.push("missing_action_id");
  if (!action.clientActionId || action.clientActionId !== utterance.clientActionId) blockers.push("client_action_id_mismatch");
  if (action.actorId !== utterance.characterId) blockers.push("actor_character_mismatch");
  if (!action.kind.trim()) blockers.push("missing_action_kind");
  if (!action.targetId.trim()) blockers.push("missing_action_target");
  if (!world.locations.includes(action.location)) blockers.push("action_location_not_in_world");
  return { actionId: action.id, eligible: blockers.length === 0, blockers, evidenceIds: [] };
}
