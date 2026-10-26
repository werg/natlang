---
description: Build assignments by capability coverage, accepted availability, consent and trainee supervision.
args: { mission: MissionDossier; change: MissionChange; requirements: MissionRequirement[]; capabilities: VolunteerCapability[] }
returns: VolunteerTeam
---
Call `measureRequirementCoverage(mission, requirements, capabilities, change)`. Require each role's verified qualification, availability, practical access condition and consent for the actual assignment. If a role is uncovered, report the exact missing capability rather than accepting a larger team. Preserve a trainee-to-qualified-supervisor edge for the same assignment window.
