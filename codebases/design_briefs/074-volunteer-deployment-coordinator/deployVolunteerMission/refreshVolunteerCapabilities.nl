---
description: Recheck volunteer qualifications at the actual assignment date and retain unrelated current capabilities.
args: { mission: MissionDossier; change: MissionChange }
returns: VolunteerCapability[]
---
Call `qualifyAtMissionWindow(mission, change)` for credentials with validity periods. Keep stated skills distinct from verified qualifications. An expired credential disqualifies only requirements that require it. Preserve withdrawn volunteers as unavailable for the affected window without deleting their prior records.
