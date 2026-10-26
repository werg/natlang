import type { EscapeRoomState, PuzzleStructureAudit } from "../../types.js";

export default function inspectPuzzleStructure(room: EscapeRoomState): PuzzleStructureAudit {
  const counts = new Map<string, number>();
  for (const node of room.nodes) counts.set(node.id, (counts.get(node.id) ?? 0) + 1);
  const duplicateNodeIds = [...counts].filter(([, count]) => count > 1).map(([id]) => id).sort();
  const knownNodes = new Set(room.nodes.map(node => node.id));
  const invalidReferences: string[] = [];
  if (!knownNodes.has(room.startNodeId)) invalidReferences.push(`missing_start:${room.startNodeId}`);
  for (const id of room.goalNodeIds) if (!knownNodes.has(id)) invalidReferences.push(`missing_goal:${id}`);
  for (const node of room.nodes) {
    for (const id of node.requires) if (!knownNodes.has(id)) invalidReferences.push(`missing_prerequisite:${node.id}:${id}`);
    for (const id of node.solves) if (!knownNodes.has(id)) invalidReferences.push(`missing_unlocked_node:${node.id}:${id}`);
  }
  const propById = new Map(room.props.map(prop => [prop.id, prop]));
  const propCoverageGaps = new Set<string>();
  for (const node of room.nodes) for (const propId of node.propIds) {
    const prop = propById.get(propId);
    if (!prop) propCoverageGaps.add(`${node.id}:${propId}:not-in-room-inventory`);
    else if (!prop.present || !prop.working || !prop.safeMechanism)
      propCoverageGaps.add(`${node.id}:${propId}:unavailable-or-unsafe`);
  }
  return {
    roomId: room.id,
    revision: room.revision,
    validNodeIds: [...knownNodes].sort(),
    duplicateNodeIds,
    invalidReferences: [...new Set(invalidReferences)].sort(),
    propCoverageGaps: [...propCoverageGaps].sort(),
  };
}
