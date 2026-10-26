---
description: Match a volunteer mission's capability coverage with verified qualifications, availability, consent and supervision.
args: { mission: MissionDossier; change: MissionChange; coordinator: DeploymentCoordinator }
returns: DeploymentProposal
---
Read the mission as requirements to cover, not a target headcount. Preserve volunteer withdrawals and qualification source versions.

Bind requirements = `interpretMissionRequirements(mission, change)` and bind each required skill, access condition, mission window and completion evidence. Do not make safety-sensitive requirements optional to fill the roster.

Bind capabilities = `refreshVolunteerCapabilities(mission, change)`. Expired or unverified qualifications fail only the role that requires them. Keep other current capabilities intact.

Bind team = `constructCapabilityTeam(mission, change, requirements, capabilities)`. Cover each required role with a qualified, available volunteer who consented to the actual assignment and mode. Remote-to-on-site changes require fresh consent and a practical-constraint check. Preserve trainee supervision in the same assignment window.

Bind briefings = `projectRoleBriefings(mission, change, team)`. Disclose private access information only to the accepted role and only during its assignment window. Require designated acknowledgment before marking a briefing ready.

Bind assignments = `replanMissionAssignments(mission, change, team, briefings)`. After a withdrawal or expiry, recompute capability coverage instead of merely replacing headcount. Keep uncovered roles and incomplete briefings as blockers.

Return `DeploymentProposal` for `coordinator.id` with the coverage map, supervision pairs, limited briefing projections and complete resume state. A roster proposal does not dispatch volunteers or claim task completion.
