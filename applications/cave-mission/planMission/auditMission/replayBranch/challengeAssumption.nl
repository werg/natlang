---
description: Independently rejudge one action against its world and carried-state premises.
args:
  action: MissionAction
  robot: RobotState
  world: WorldModel
  survey: SurveyEvidence
returns: CrossingDecision
---
1. Select the original source clauses about this action's direction, location, gear, sample and sensor
   scope. Compare them to the reconstructed world entry rather than trusting a prior allowed flag.
2. In eval declare a local ActionChallenge with allowed:boolean, reason:string and witnesses:Witness[].
   Create a scoped NL literal capturing the selected clauses, world facts and current gear. Directly
   await it on the proposed action, asking for a concrete rule violation or source-backed permission.
3. Consume the returned permission. Insufficient energy, wrong direction, full-carrier tunnel use,
   rerigging a consumed spool or a pickup outside the sample chamber returns no nextRobot.
4. If permitted, calculate the exact nextRobot: subtract cost; update location only for move, rope/
   rigSide only for rig, sampleHeld only for pickup, and no physical field beyond energy for observe.
5. A blind anchor may be rigged physically but cannot satisfy a later V-to-J ascent. Require the
   world's correctRigSide for that ascent. Observation cannot retroactively move a wrongly fixed rope.
6. Return worldId, allowed, nextRobot, reason and witnesses. This independent judgment is consumed
   by replayBranch; recording an unused challenge string is not an audited state transition.
