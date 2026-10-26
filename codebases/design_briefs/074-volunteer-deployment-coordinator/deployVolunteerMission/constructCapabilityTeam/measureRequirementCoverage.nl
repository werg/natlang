---
description: Match qualified and consenting volunteers to required mission roles without using headcount as a proxy.
args: { mission: MissionDossier; requirements: MissionRequirement[]; capabilities: VolunteerCapability[]; change: MissionChange }
returns: CapabilityCoverage[]
---
For each requirement, list only volunteers with a current verified qualification, a nonconflicting available window and consent to the role's location and mode. Detect withdrawal, expired credential and consent scope separately. If the role is trainee-designated, require a named qualified supervisor with overlapping availability and accepted assignment.
