---
description: Freeze affected show activity under its approved safety protocol and require exact all-clear authority.
args: { incident: ShowIncident; show: LiveShowState; event: TheatreEvent }
returns: ContinuityReplay
---
1. Collect the show production, script and safety sources together with the event evidence and updates. Bind procedure = matchStopProcedure(incident, evidence, show.safetyRules), then frozen = freezeAffectedCues(incident, show.production, show.checkpoint). These operations interpret an existing stop protocol; they do not operate a physical interlock.
2. Bind proposedCheckpoint as the complete prior checkpoint with state held and safetyHold true. Preserve its production, version, route, all executed receipt IDs, audience and character knowledge, resource reservations and prior history. Retain all prior holdSignalIds and add the actual new stop signal once. Add frozen.frozenCueIds to pendingCueIds without duplication.
3. Return ContinuityReplay { sourceCheckpoint: show.checkpoint.id, newCheckpointId: "", actualPath: frozen.alreadyExecutedIds, unobservedPlans: frozen.frozenCueIds, knowledgeByAudienceGroup: show.checkpoint.audienceKnowledgeByGroup, safetyHold: true, holdSignalIds: proposedCheckpoint.holdSignalIds, resetTasks: [], blockers: procedure.blockers plus the exact named-lead all-clear still required, proposedCheckpoint }.
4. A missing procedure or lead remains explicit. The proposed held checkpoint does not claim the physical stop was delivered; that requires the production's independent safety system and actual acknowledgment. A generated plan can never clear the hold.
