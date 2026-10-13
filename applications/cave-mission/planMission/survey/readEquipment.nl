---
description: Recover resource transitions and the distinct scopes of available observations.
args:
  clauses: SourceClause[]
returns: EquipmentSurvey
---
1. Inventory the initial carried objects and source-authorized changes: rigging, sample pickup,
   consumption, irreversibility, and any explicitly allowed recovery or transfer.
2. Return GearRule entries for conditions and consequences. Keep the empty/full carrier distinction
   separate from rope availability; carrying a sample and rigging a rope change different state fields.
3. Preserve prohibitions on discarding, opening or transferring the sealed sample. A route that
   returns the robot but loses the task object cannot satisfy the mission goal.
4. For each sensor, record its ID, available location, cost, source-listed answer categories and scope.
   State whether it sees a surface feature, deep structure or another property of the physical world.
5. Distinguish an answer's meaning from the observation action's physical effect. Observing can
   refine belief without moving the robot, changing the map or restoring a resource.
6. Record repeated-query cost and consistency if stated. An identical answer under every supported
   map is still an observation, but it cannot settle a route-relevant frame ambiguity.
7. Preserve any unresolved cost or scope clause explicitly. Do not invent a more capable sensor or
   a free observation to make a policy feasible. Return gear, sensors and unresolved.
