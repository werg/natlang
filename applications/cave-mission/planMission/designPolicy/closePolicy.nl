---
description: Project winning decisions into a complete rooted observation policy without changing them.
args:
  graph: BeliefGraph
  values: StateValue[]
  request: MissionRequest
returns: BeliefPolicy
---
1. Find the value of graph.initialStateId. If it is absent or losing, return null rootNodeId, no nodes,
   null objective values and the explicit blocking reasons. Do not output a lucky single-world route.
2. Traverse the finite reachable chosen-action graph beginning at the initial state. Create one
   PolicyNode per reached state and preserve its chosen MissionAction and every outcome successor.
3. For a goal state, create a terminal node with null action and no outcomes. For a nonterminal,
   use exactly the action selected by its StateValue; do not improvise a route during serialization.
4. Map state IDs to arbitrary unique node IDs. Preserve each outcome's answer, next-node reference
   and world-ID set. This reference renaming does not change what the policy does in any world.
5. Shared successor states may share a node; branch order is irrelevant. Never omit a less attractive
   observation answer or collapse two cells that require different gear commitments.
6. Return the policy with worstCaseEnergy and worstCaseObservations from the initial value and no
   blocked reasons. Its all-world claims remain subject to auditMission's separate physical replay.
