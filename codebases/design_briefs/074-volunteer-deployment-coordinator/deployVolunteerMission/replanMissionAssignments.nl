---
description: Recompute the affected capability coverage when a volunteer withdraws, qualification expires or work mode changes.
args: { mission: MissionDossier; change: MissionChange; team: VolunteerTeam; briefings: BriefingProjection[] }
returns: VolunteerRevision
---
Call `checkAffectedAssignments(mission, change, team)` again after applying only the current withdrawal, expiry or mode delta. Preserve the prior team version and invalidate only assignments whose qualification, availability, consent, supervisor or access briefing dependency changed. Keep uncovered requirements visible for coordinator escalation.
