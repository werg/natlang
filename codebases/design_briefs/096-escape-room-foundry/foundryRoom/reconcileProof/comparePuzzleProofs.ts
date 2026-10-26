import type { PuzzleProof, PuzzleVerification, PuzzleProofCheck } from "../../types.js";

const sameList = (left: string[], right: string[]) =>
  JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());

export default function comparePuzzleProofs(
  computed: PuzzleVerification,
  supplied: PuzzleProof,
): PuzzleProofCheck {
  const differences: string[] = [];
  if (supplied.graphRevision !== computed.graphRevision) differences.push("graphRevision");
  if (!sameList(supplied.reachableNodeIds, computed.reachableNodeIds)) differences.push("reachableNodeIds");
  if (!sameList(supplied.unreachableGoalIds, computed.unreachableGoalIds)) differences.push("unreachableGoalIds");
  if (!sameList(supplied.cycleNodeIds, computed.cycleNodeIds)) differences.push("cycleNodeIds");
  if (!sameList(supplied.softLockStates, computed.softLockStates)) differences.push("softLockStates");
  if (supplied.exitInvariantHolds !== computed.exitInvariantHolds) differences.push("exitInvariantHolds");
  if (!sameList(supplied.propCoverageGaps, computed.propCoverageGaps)) differences.push("propCoverageGaps");
  return { matches: differences.length === 0, differences };
}
