---
description: Execute a proposed policy as pure state transitions in one declared world.
args:
  world: WorldModel
  request: MissionRequest
  survey: SurveyEvidence
  policy: BeliefPolicy
returns: WorldReplay
---
1. Start with the source initial RobotState and policy.rootNodeId. Keep visited node IDs, action IDs,
   total energy and observation count. Every executed action has a positive source cost.
2. Follow the finite policy nodes until a terminal is reached. A repeated node, missing next node,
   unsupported action or wrong belief membership is a defect and stops this world's replay explicitly.
3. For each action, call challengeAssumption(action, currentRobot, world, survey). Consume the
   returned CrossingDecision; its nextRobot is the only state change this replay may apply.
4. For an observation, derive the source answer for this actual world and choose the matching edge.
   Require exactly one such edge whose world set contains world.worldId. The action changes energy only.
5. For a deterministic action, require one null-answer edge containing this world. Apply its physical
   delta: movement, one-use rigging or sample pickup. Check direction, clearance and protected sample rules.
6. Append the actual node/action references and update exact totals. A terminal succeeds only when
   location, sample possession and remaining energy satisfy the mission goal.
7. Return WorldReplay with the final state and findings. Do not repair an unsafe policy during replay;
   preserve its first counterexample so the planner's proposed artifact can be corrected and remeasured.
